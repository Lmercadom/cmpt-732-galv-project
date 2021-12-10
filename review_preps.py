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



stars_conveter = udf(lambda x : convert_rating(x))
text_cleaner = udf(lambda x : clean_text(x))
def main(reviews_file):
    # 1) Get relevant columns from review data
    reviews_df = spark.read.parquet(reviews_file).select('review_id','contents', 'stars').cache()
    # print(reviews_df.collect()[0][1])
    # print()
    # print(reviews_df.collect()[1][1])

    #2) Filter out non English Reviews
    # reviews_df = reviews_df.withColumn('language', get_language(reviews_df['contents']))
    # reviews_df = reviews_df.filter(reviews_df['language'] == 'en')

    #3) Convert star reviews into positive or negative
    reviews_df = reviews_df.select('review_id', 'contents', stars_conveter(reviews_df['stars']).alias('sentiment'))
        # print(new_ratings.collect()[0][2])
    # new_ratings.show()

    # 4 PROCESS REVIEW TEXT
    # 4.1 Remove Symbols, convert to lowercase & Remove Stop words

    reviews_clean = reviews_df.select('review_id', text_cleaner(reviews_df['contents']).alias('contents'), 'sentiment')
    # print(reviews_clean.collect()[0][1])
    # print()
    # print(reviews_clean.collect()[0][2])




    #5 Combined the text of all reviews into a single one gruped by sentiment
    combined_reviews = reviews_clean.groupBy('sentiment').agg(concat_ws(" ", collect_list("contents")).alias('new_text'))
    # print(combined_reviews.collect()[1][1])
    print(combined_reviews.show())

    # print(combined_reviews.show())
    # print('NEGATIVE REVIEWS')
    # print()
    # print(combined_reviews.collect()[0][1])
    # print()
    # print('POSITIVE REVIEWS')
    # print(combined_reviews.collect()[1][1])

    #6) Calculate frequency for positive and negative reviews
    # a) break text into individual words
    # b)
    combined_reviews = combined_reviews.withColumn('word_items', functions.split(combined_reviews['new_text'], " "))
    reviews_words = combined_reviews.select(combined_reviews['sentiment'], functions.explode(combined_reviews['word_items']).alias('word_items'))
    count_table = reviews_words.groupBy('sentiment','word_items').count()
    frequency_table = count_table.sort(count_table['count'].desc())
    # frequency_table = count_table.withColumn('frequency', functions.col('count') / total)
    negative_reviews = frequency_table.filter(frequency_table['sentiment'] == 0)
    positive_reviews = frequency_table.filter(frequency_table['sentiment'] == 1)
    print(negative_reviews.show(5))
    print(positive_reviews.show(5))
    # Save both files as positive and negative_reviews


    # frequency.plot()











if __name__ == '__main__':
    reviews_file = sys.argv[1]
    spark = SparkSession.builder.appName('Reviews Parquet').getOrCreate()
    assert spark.version >= '3.0' # make sure we have Spark 3.0+
    spark.sparkContext.setLogLevel('WARN')
    sc = spark.sparkContext
    main(reviews_file)
