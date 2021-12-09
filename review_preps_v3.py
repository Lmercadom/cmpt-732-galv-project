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
from pyspark.ml.feature import NGram
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from nltk.tokenize import sent_tokenize
from afinn import Afinn
import sparknlp
# import sparknlp

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

def remove_punct(text):
    nopunct = regex.sub(" ", text)
    return nopunct

def remove_stopping_words(text, stopwords=[]):
    word_tokens = word_tokenize(text)
    filtered_sentence = [w for w in word_tokens if not w.lower() in stopwords]
    new_sentence = ' '.join(filtered_sentence)
    # print(word_tokens)
    # print(filtered_sentence)
    return new_sentence


def clean_text(text):
    # Remove some symbols & punctuation
    regex = re.compile('[' + re.escape(string.punctuation) + '0-9\\r\\t\\n]')
    nopunct = regex.sub(" ", text)

    # Remove stopping words & convertert text to lowercase
    word_tokens = word_tokenize(nopunct)
    filtered_sentence = [w.lower() for w in word_tokens if not w.lower() in stop_words]
    new_sentence = ' '.join(filtered_sentence)
    # print(word_tokens)
    # print(filtered_sentence)
    return new_sentence

def tokenize_text(text):
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
text_cleaner = udf(lambda x : clean_text(x))
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
    reviews_df.show(5)
    print(reviews_df.printSchema())

    # 5) Assign positive, neutral and negative sentiment to each sentence
    sentence_table = reviews_df.select(reviews_df['rating'], functions.explode(reviews_df['sentence_items']).alias('sentences')).cache()
    sentence_table = sentence_table.withColumn('sentiment_score', get_sentence_sentiments(sentence_table['sentences']))
    sentence_table.show(5)
    print(sentence_table.printSchema())

    #6) 4.1 Remove Symbols, convert to lowercase & Remove Stop words
    # sentences_clean = sentence_table.withColumn('sentence_clean', text_cleaner(sentence_table['sentences']))

    # Clean sentences using spark nlp and pipeline features
    # from pyspark.nlp import DocumentAssembler
    # documentAssembler = DocumentAssembler() \
    #                 .setInputCol("sentences") \
    #                 .setOutputCol("document") \

    from pyspark.ml.feature import Tokenizer
    tokenizer = Tokenizer() \
    .setInputCol('sentences') \
    .setOutputCol('tokenized')



    # from pyspark.ml.features import Normalizer
    # normalizer = Normalizer() \
    # .setInputCols(['tokenized']) \
    # .setOutputCol('normalized') \
    # .setLowercase(True)


    # from sparknlp.annotator import LemmatizerModel
    # lemmatizer = LemmatizerModel.pretrained() \
    # .setInputCols(['normalized']) \
    # .setOutputCol('lemmatized')

    from pyspark.ml.feature import StopWordsRemover
    stopwords_cleaner = StopWordsRemover() \
    .setInputCol('tokenized') \
    .setOutputCol('no_stop_tokenized') \
    .setStopWords(stop_words)

    n_grams_generator = NGram(n=int(ngrams)).setInputCol('no_stop_tokenized').setOutputCol("nGrams")
    from pyspark.ml import Pipeline
    pipeline = Pipeline() \
    .setStages([
             tokenizer,
             stopwords_cleaner,
             n_grams_generator])
    output_table = pipeline.fit(sentence_table).transform(sentence_table)
    output_table.show(5)

    ngrams_table = output_table.select(output_table['rating'], output_table['sentiment_score'], functions.explode(output_table['nGrams']).alias('nGrams')).cache()
    ngrams_table.show(5)

    n_grams_count = ngrams_table.groupBy('rating', 'sentiment_score', 'nGrams').count()
    n_grams_count = n_grams_count.sort(n_grams_count['count'].desc())
    n_grams_count.show(5)

    negative_reviews_pos = n_grams_count.filter((n_grams_count['rating'] == 0) & (n_grams_count['sentiment_score'] == 1))
    negative_reviews_neg = n_grams_count.filter((n_grams_count['rating'] == 0) & (n_grams_count['sentiment_score'] == 0))
    # positive_reviews = n_grams_count.filter(n_grams_count['sentiment'] == 1)
    # negative_reviews = n_grams_count.filter(n_grams_count['rating'] == 0)
    # print(negative_reviews.show(5))
    # print(positive_reviews.show(5))


    # Select top 20 words in each category
    # Negative Reviews

    top20_neg_pos = spark.createDataFrame(negative_reviews_pos.head(20))
    # top20_neg.show()
    top20_neg_pos = top20_neg_pos.select(top20_neg_pos['nGrams'], top20_neg_pos['count']).toPandas()
    plt.figure(figsize=(10, 5))
    plt.barh(top20_neg_pos['nGrams'], top20_neg_pos['count'])
    plt.savefig(outdir + f'/negative_reviews_histogram_pos_{ngrams}')
    print(top20_neg_pos.head())


    plt.clf()
    plt.figure(figsize=(10, 5))
    top20_neg_neg= spark.createDataFrame(negative_reviews_neg.head(20))
    top20_neg_neg = top20_neg_neg.select(top20_neg_neg['nGrams'], top20_neg_neg['count']).toPandas()
    print(top20_neg_neg.head())
    plt.barh(top20_neg_neg['nGrams'], top20_neg_neg['count'])
    plt.savefig(outdir + f'/negative_reviews_histogram_neg_{ngrams}')

    # Positive
    positive_reviews_pos = n_grams_count.filter((n_grams_count['rating'] == 1) & (n_grams_count['sentiment_score'] == 1))
    positive_reviews_neg = n_grams_count.filter((n_grams_count['rating'] == 1) & (n_grams_count['sentiment_score'] == 0))

    top20_pos_pos = spark.createDataFrame(positive_reviews_pos.head(20))
    # top20_neg.show()
    top20_pos_pos = top20_pos_pos.select(top20_pos_pos['nGrams'], top20_pos_pos['count']).toPandas()
    plt.figure(figsize=(10, 5))
    plt.barh(top20_pos_pos['nGrams'], top20_pos_pos['count'])
    plt.savefig(outdir + f'/positive_reviews_histogram_pos_{ngrams}_{business_id}')
    print(top20_pos_pos.head())


    plt.clf()
    plt.figure(figsize=(10, 5))
    top20_pos_neg= spark.createDataFrame(positive_reviews_neg.head(20))
    top20_pos_neg = top20_pos_neg.select(top20_pos_neg['nGrams'], top20_pos_neg['count']).toPandas()
    print(top20_pos_neg.head())
    plt.barh(top20_pos_neg['nGrams'], top20_pos_neg['count'])
    plt.savefig(outdir + f'/positive_reviews_histogram_neg_{ngrams}_{business_id}')






    # # 4 PROCESS REVIEW TEXT
    # # 4.1 Remove Symbols, convert to lowercase & Remove Stop words
    #
    # reviews_clean = reviews_df.select('review_id', text_cleaner(reviews_df['contents']).alias('contents'), 'sentiment')
    # # print(reviews_clean.collect()[0][1])
    # # print()
    # # print(reviews_clean.collect()[0][2])
    #
    #
    #
    #
    # #5 Combined the text of all reviews into a single one gruped by sentiment
    # combined_reviews = reviews_clean.groupBy('sentiment').agg(concat_ws(" ", collect_list("contents")).alias('new_text'))
    # # print(combined_reviews.collect()[1][1])
    # # print(combined_reviews.show())
    #
    # # print(combined_reviews.show())
    # # print('NEGATIVE REVIEWS')
    # # print()
    # # print(combined_reviews.collect()[0][1])
    # # print()
    # # print('POSITIVE REVIEWS')
    # # print(combined_reviews.collect()[1][1])
    #
    # #6) Calculate frequency for positive and negative reviews
    # # a) break text into individual words
    # # b)
    # combined_reviews = combined_reviews.withColumn('word_items', functions.split(combined_reviews['new_text'], " "))
    # # reviews_words = combined_reviews.select(combined_reviews['sentiment'], functions.explode(combined_reviews['word_items']).alias('word_items'))
    # # count_table = reviews_words.groupBy('sentiment','word_items').count()
    # # frequency_table = count_table.sort(count_table['count'].desc())
    # # # frequency_table = count_table.withColumn('frequency', functions.col('count') / total)
    # # negative_reviews = frequency_table.filter(frequency_table['sentiment'] == 0)
    # # positive_reviews = frequency_table.filter(frequency_table['sentiment'] == 1)
    # # print(negative_reviews.show(5))
    # # print(positive_reviews.show(5))
    # # # Save both files as positive and negative_reviews
    #
    #
    # # frequency.plot()
    # # Create NGRAMS
    # # n_grams = NGram(n=1)
    # # n_grams.setInputCol("word_items")
    # # n_grams.setOutputCol("nGrams")
    # # ngrams_table = n_grams.transform(combined_reviews)
    # # ngrams_table = ngrams_table.select(ngrams_table['sentiment'], functions.explode(ngrams_table['nGrams']).alias('nGrams_items'))
    # # # ngrams_table.show(5)
    # # n_grams_count = ngrams_table.groupBy('sentiment','nGrams_items').count()
    # # n_grams_count = n_grams_count.sort(n_grams_count['count'].desc())
    # # # n_grams_count.show(5)
    # # negative_reviews = n_grams_count.filter(n_grams_count['sentiment'] == 0)
    # # positive_reviews = n_grams_count.filter(n_grams_count['sentiment'] == 1)
    # # # print(negative_reviews.show(5))
    # # # print(positive_reviews.show(5))
    #
    #
    # n_grams = NGram(n=int(ngrams))
    # n_grams.setInputCol("word_items")
    # n_grams.setOutputCol("nGrams")
    # ngrams_table = n_grams.transform(combined_reviews)
    # ngrams_table = ngrams_table.select(ngrams_table['sentiment'], functions.explode(ngrams_table['nGrams']).alias('nGrams_items'))
    # # ngrams_table.show(5)
    # n_grams_count = ngrams_table.groupBy('sentiment','nGrams_items').count()
    # n_grams_count = n_grams_count.sort(n_grams_count['count'].desc())
    # # n_grams_count.show(5)
    # negative_reviews = n_grams_count.filter(n_grams_count['sentiment'] == 0)
    # positive_reviews = n_grams_count.filter(n_grams_count['sentiment'] == 1)
    # # print(negative_reviews.show(5))
    # # print(positive_reviews.show(5))
    #
    #
    # # Select top 20 words in each category
    # top20_neg = spark.createDataFrame(negative_reviews.head(20))
    # # top20_neg.show()
    # top20_neg = top20_neg.select(top20_neg['nGrams_items'], top20_neg['count']).toPandas()
    # plt.figure(figsize=(10, 5))
    # plt.barh(top20_neg['nGrams_items'], top20_neg['count'])
    # plt.savefig(outdir + f'/negative_reviews_histogram_{ngrams}')
    # print(top20_neg.head())
    #
    # plt.clf()
    # plt.figure(figsize=(10, 5))
    # top20_pos= spark.createDataFrame(positive_reviews.head(20))
    # top20_pos = top20_pos.select(top20_pos['nGrams_items'], top20_pos['count']).toPandas()
    # print(top20_pos.head())
    # plt.barh(top20_pos['nGrams_items'], top20_pos['count'])
    # plt.savefig(outdir + f'/positive_reviews_histogram_{ngrams}')







if __name__ == '__main__':
    reviews_file = sys.argv[1]
    business_id = sys.argv[2]
    ngrams = sys.argv[3]
    outdir = sys.argv[4]
    spark = SparkSession.builder.appName('Reviews Parquet').getOrCreate()
    assert spark.version >= '3.0' # make sure we have Spark 3.0+
    spark.sparkContext.setLogLevel('WARN')
    sc = spark.sparkContext
    main(reviews_file, business_id, ngrams, outdir)
