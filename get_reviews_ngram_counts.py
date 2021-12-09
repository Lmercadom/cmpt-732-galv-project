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
from nltk.tokenize import word_tokenize
from nltk import FreqDist
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from nltk.tokenize import sent_tokenize
from afinn import Afinn
import sparknlp
from sparknlp.base import DocumentAssembler, Finisher
from sparknlp.annotator import Tokenizer, Normalizer, LemmatizerModel, StopWordsCleaner, NGramGenerator
from pyspark.ml import Pipeline

af = Afinn()

stop_words = stopwords.words('english')
regex = re.compile('[' + re.escape(string.punctuation) + '0-9\\r\\t\\n]')

def convert_rating(rating):
    # Everything greater or equal to 4 is "positive"
    # Else " Negative"
    if rating >=4:
        return 1
    else:
        return 0

def tokenize_text(text):
    # break down reviews into sentences and join them with ;
    sentence_tokens = sent_tokenize(text)
    sentence_tokens = ';'.join(sentence_tokens)
    return sentence_tokens

def assign_sentiment(sentence):
    sentiment_score = af.score(sentence)
    if sentiment_score > 0:
        return 1
    elif sentiment_score < 0:
        return -1
    else:
        return 0


stars_conveter = udf(lambda x : convert_rating(x))
get_language = udf(lambda x : detect(x))
get_sentence_tokens = udf(lambda x: tokenize_text(x))
get_sentence_sentiments = udf(lambda x: assign_sentiment(x))

def main(reviews_file, business_id, ngrams, outdir):
    # 1) Get relevant columns from review data
    reviews_df = spark.read.parquet(reviews_file).select('review_id','contents', 'stars', 'business_id')
    reviews_df = reviews_df.filter(reviews_df['business_id'] == str(business_id))

    #2) Filter out non English Reviews
    reviews_df = reviews_df.withColumn('language', get_language(reviews_df['contents']))
    reviews_df = reviews_df.filter(reviews_df['language'] == 'en')

    #3) Convert star reviews into positive and not negative reviews
    #   stars >= 4 are good, stars < 4 are not good
    reviews_df = reviews_df.select('review_id', 'contents', stars_conveter(reviews_df['stars']).alias('rating')).cache()
    # print(reviews_df.groupBy('sentiment').count().show(5))

    #4) Break down reviews into sentences
    reviews_df = reviews_df.withColumn('sentences', get_sentence_tokens(reviews_df['contents']))
    reviews_df = reviews_df.withColumn('sentence_items', functions.split(reviews_df['sentences'], ";"))

    # 5) Assign positive, neutral and negative sentiment to each sentence
    sentence_table = reviews_df.select(reviews_df['rating'], functions.explode(reviews_df['sentence_items']).alias('sentences')).cache()
    sentence_table = sentence_table.withColumn('sentiment_score', get_sentence_sentiments(sentence_table['sentences']))


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


    # .setCleanupPatterns
    # Remove symbols and character
    normalizer = Normalizer() \
    .setInputCols(['tokenized']) \
    .setOutputCol('normalized') \
    .setLowercase(True)


    # Create base froms of words so that different forms of the same word
    # are grouped as one
    lemmatizer = LemmatizerModel.pretrained() \
    .setInputCols(['normalized']) \
    .setOutputCol('lemmatized')

    # Clean stop words
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

    output_table = pipeline.fit(sentence_table).transform(sentence_table)

    ngrams_table = output_table.select(output_table['rating'], output_table['sentiment_score'], functions.explode(output_table['finished_nGrams']).alias('nGrams')).cache()


    print('Generating ngram count table...')
    n_grams_count = ngrams_table.groupBy('rating', 'sentiment_score', 'nGrams').count()
    n_grams_count = n_grams_count.sort(n_grams_count['count'].desc())


    print('Generating plots...')
    # Select top 20 ngrams in each category
    # Negative Reviews

    # ngrams in negative reviews coming from sentences that have a positive sentiment
    negative_reviews_pos = n_grams_count.filter((n_grams_count['rating'] == 0) & (n_grams_count['sentiment_score'] == 1))
    # ngrams in negative reviews coming from sentences that have a negative sentiment
    negative_reviews_neg = n_grams_count.filter((n_grams_count['rating'] == 0) & (n_grams_count['sentiment_score'] == 0))

    top20_neg_pos = spark.createDataFrame(negative_reviews_pos.head(20))
    # top20_neg.show()
    top20_neg_pos = top20_neg_pos.select(top20_neg_pos['nGrams'], top20_neg_pos['count']).toPandas()
    plt.figure(figsize=(10, 5))
    plt.barh(top20_neg_pos['nGrams'], top20_neg_pos['count'])
    plt.savefig(outdir + f'/{business_id}_negative_reviews_histogram_pos_{ngrams}')
    print(top20_neg_pos.head())


    plt.clf()
    plt.figure(figsize=(10, 5))
    top20_neg_neg= spark.createDataFrame(negative_reviews_neg.head(20))
    top20_neg_neg = top20_neg_neg.select(top20_neg_neg['nGrams'], top20_neg_neg['count']).toPandas()
    print(top20_neg_neg.head())
    plt.barh(top20_neg_neg['nGrams'], top20_neg_neg['count'])
    plt.savefig(outdir + f'/{business_id}_negative_reviews_histogram_neg_{ngrams}')



    # Positive Reviews
    # ngrams in positive reviews coming from sentences that have a positive sentiment
    positive_reviews_pos = n_grams_count.filter((n_grams_count['rating'] == 1) & (n_grams_count['sentiment_score'] == 1))
    # ngrams in positive reviews coming from sentences that have a negative sentiment
    positive_reviews_neg = n_grams_count.filter((n_grams_count['rating'] == 1) & (n_grams_count['sentiment_score'] == 0))

    top20_pos_pos = spark.createDataFrame(positive_reviews_pos.head(20))
    # top20_neg.show()
    top20_pos_pos = top20_pos_pos.select(top20_pos_pos['nGrams'], top20_pos_pos['count']).toPandas()
    plt.figure(figsize=(10, 5))
    plt.barh(top20_pos_pos['nGrams'], top20_pos_pos['count'])
    plt.savefig(outdir + f'/{business_id}_positive_reviews_histogram_pos_{ngrams}')
    print(top20_pos_pos.head())


    plt.clf()
    plt.figure(figsize=(10, 5))
    top20_pos_neg= spark.createDataFrame(positive_reviews_neg.head(20))
    top20_pos_neg = top20_pos_neg.select(top20_pos_neg['nGrams'], top20_pos_neg['count']).toPandas()
    print(top20_pos_neg.head())
    plt.barh(top20_pos_neg['nGrams'], top20_pos_neg['count'])
    plt.savefig(outdir + f'/{business_id}_positive_reviews_histogram_neg_{ngrams}')
    print('program finished')


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
