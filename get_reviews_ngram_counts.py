import sys
assert sys.version_info >= (3, 5) # make sure we have Python 3.5+

from pyspark.sql import SparkSession, functions, types
from pyspark.sql.functions import broadcast, concat_ws, collect_list, explode
from pyspark.sql.functions import udf
from langdetect import detect
import string
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize
from afinn import Afinn
import sparknlp
from sparknlp.base import DocumentAssembler, Finisher
from sparknlp.annotator import Tokenizer, Normalizer, LemmatizerModel, StopWordsCleaner, NGramGenerator
from pyspark.ml import Pipeline

af = Afinn()

stop_words = stopwords.words('english')

def convert_rating(rating):
    '''
    Converts start ratings into "positive" or "negative"
    rating >= 4 is "positive"
    rating < 4 is "negative"
    '''

    if rating >=4:
        return 1
    else:
        return 0

def tokenize_text(text):
    '''
    Breaks out review text into group of sentences
    '''
    sentence_tokens = sent_tokenize(text)
    sentence_tokens = ';'.join(sentence_tokens)
    return sentence_tokens

def assign_sentiment(sentence):
    '''
    assigns overall positive, negative and neutral sentiment scores
    to sentences based on the Affin Lexicon
    positive scores are positive sentiments
    zero scores are neutral
    negative scores are negative sentiments

    '''
    sentiment_score = af.score(sentence)
    if sentiment_score > 0:
        return 1
    elif sentiment_score < 0:
        return -1
    else:
        return 0

# convert functions to spark udf
stars_conveter = udf(lambda x : convert_rating(x))
get_language = udf(lambda x : detect(x))
get_sentence_tokens = udf(lambda x: tokenize_text(x))
get_sentence_sentiments = udf(lambda x: assign_sentiment(x))

def main(reviews_file, businesses, ngrams, outdir):
    '''
    Generates a count table for ngrams found in reviews
    grouped by business id, review ratings and sentence sentiment

    inputs:
    reviews_file: The file containing the reviews
    businesses: a txt file with businesses' ids
    ngrams: the lenght of the ngrams
    outdir: output directory

    outputs:
    table of ngrams and their frequency in csv
    '''
    # Get the business ids
    with open(businesses) as f:
        business_ids= f.read().split('\n')


    # 1) Get relevant columns from review data
    reviews_df = spark.read.parquet(reviews_file).select('review_id', \
                                                        'contents', \
                                                        'stars', \
                                                        'business_id').repartition(32)
    reviews_df = reviews_df.filter(reviews_df['business_id'].isin(business_ids))



    #2) Filter out non English Reviews
    reviews_df = reviews_df.withColumn('language', get_language(reviews_df['contents']))
    reviews_df = reviews_df.filter(reviews_df['language'] == 'en')

    #3) Convert star reviews into positive and not negative reviews
    #   stars >= 4 are good, stars < 4 are not good
    reviews_df = reviews_df.select('review_id', 'contents', \
                                    stars_conveter(reviews_df['stars']).alias('rating'),\
                                    'business_id')

    #4) Break down reviews into sentences
    reviews_df = reviews_df.withColumn('sentences', \
                                        get_sentence_tokens(reviews_df['contents']))

    reviews_df = reviews_df.withColumn('sentence_items', \
                                        functions.split(reviews_df['sentences'], ";"))

    # 5) Assign positive, neutral and negative sentiment to each sentence
    sentence_table = reviews_df.select(reviews_df['business_id'], \
                                       reviews_df['rating'], \
                                       functions.explode(reviews_df['sentence_items']).alias('sentences')).cache()

    sentence_table = sentence_table.withColumn('sentiment_score', \
                                               get_sentence_sentiments(sentence_table['sentences'])).cache()


    # Clean sentences and generate ngrams using spark nlp and pipeline features
    # transform column into format for nlp pipelines
    print('Generating ngrams...')
    documentAssembler = DocumentAssembler() \
                    .setInputCol("sentences") \
                    .setOutputCol("document") \

    # breaks up each sentence into words
    tokenizer = Tokenizer() \
    .setInputCols(['document']) \
    .setOutputCol('tokenized')


    # Remove symbols and characters
    normalizer = Normalizer() \
    .setInputCols(['tokenized']) \
    .setOutputCol('normalized') \
    .setLowercase(True)


    # Create base froms of words so that different forms of the same word
    # are grouped as one single word
    lemmatizer = LemmatizerModel.pretrained() \
    .setInputCols(['normalized']) \
    .setOutputCol('lemmatized')

    # Clean stop words (common english word)
    stopwords_cleaner = StopWordsCleaner() \
    .setInputCols(['lemmatized']) \
    .setOutputCol('no_stop_lemmatize') \
    .setStopWords(stop_words)

    # Generate ngrams of length N
    n_grams_generator = NGramGenerator() \
    .setInputCols(['no_stop_lemmatize']) \
    .setOutputCol("nGrams") \
    .setN(int(ngrams))


    # convert token column into human readable form
    finisher = Finisher() \
     .setInputCols(['nGrams'])

    # Create pipeline
    pipeline = Pipeline() \
    .setStages([
             documentAssembler,
             tokenizer,
             normalizer,
             lemmatizer,
             stopwords_cleaner,
             n_grams_generator,
             finisher])

    # Run Pipeline
    output_table = pipeline.fit(sentence_table).transform(sentence_table).cache()

    # create a row for each ngram
    ngrams_table = output_table.select(output_table['business_id'], \
                                       output_table['rating'], \
                                       output_table['sentiment_score'], \
                                       functions.explode(output_table['finished_nGrams']).alias('nGrams')).cache()

    print('Generating ngram count table...')
    n_grams_count = ngrams_table.groupBy('business_id', 'rating', 'sentiment_score', 'nGrams').count()
    n_grams_count = n_grams_count.coalesce(1)

    n_grams_count.write.mode("overwrite").csv(outdir + '/n_grams_count.csv', header='true')

if __name__ == '__main__':
    reviews_file = sys.argv[1]
    business_id = sys.argv[2]
    ngrams = sys.argv[3]
    outdir = sys.argv[4]
    spark = SparkSession.builder.appName('Reviews Parquet').config("com.johnsnowlabs.nlp:spark-nlp_2.12:3.3.4").getOrCreate()
    assert spark.version >= '3.0' # make sure we have Spark 3.0+
    spark.sparkContext.setLogLevel('WARN')
    sc = spark.sparkContext
    main(reviews_file, business_id, ngrams, outdir)
