import sys
assert sys.version_info >= (3, 5) # make sure we have Python 3.5+

from pyspark.sql import SparkSession, functions, types
from pyspark.sql.functions import broadcast
from pyspark.sql.functions import udf
from langdetect import detect

# get_language = udf(lambda x : detect(x))
def get_language(x):
    try:
        detect(x)
    except:
        return '--'
lang_identifier = udf(lambda x: get_language(x))
def main(reviews_file):
    reviews_schema = types.StructType([
    types.StructField('review_id', types.StringType()),
    types.StructField('user_id', types.StringType()),
    types.StructField('business_id', types.StringType()),
    types.StructField('stars', types.FloatType()),
    types.StructField('useful', types.IntegerType()),
    types.StructField('funny', types.IntegerType()),
    types.StructField('cool', types.IntegerType()),
    types.StructField('contents', types.StringType()),
    types.StructField('date', types.TimestampType()),
    ])

    # PREPROCESSING REVIEWS
    reviews_df = spark.read.parquet(reviews_file).select('review_id','contents', 'stars').repartition(16)
    # reviews_df.show(5)

    # 1) FILTER OUT NON ENGLISH REVIEWS
    # Check if Reviews are only in English.
    # Find non-english reviews

    reviews_df = reviews_df.withColumn('language', lang_identifier(reviews_df['contents']))
    # print(reviews_df.show())
    # all_languages = reviews_df.groupBy('language').count()
    # print(all_languages.show(5))
    reviews_df = reviews_df.fliter(reviews_df['language'] != 'en')

    # 2) REMOVE PUNCTUATION MARKS & OTHER SYMBOLS


    # 3) CONVERT REVIEW RATING INTO EITHER POSITIVE OR NEGATIVE


    #4) REMOVE STOP WORDS


    #5) CREATE N-GRAMS OF WORDS




if __name__ == '__main__':
    reviews_file = sys.argv[1]
    spark = SparkSession.builder.appName('Reviews Parquet').getOrCreate()
    assert spark.version >= '3.0' # make sure we have Spark 3.0+
    spark.sparkContext.setLogLevel('WARN')
    sc = spark.sparkContext
    main(reviews_file)
