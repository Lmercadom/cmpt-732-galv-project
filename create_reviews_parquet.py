import sys
assert sys.version_info >= (3, 5) # make sure we have Python 3.5+

from pyspark.sql import SparkSession, functions, types
from pyspark.sql.functions import broadcast

def main(business_file, reviews_file, outdir):
    reviews_schema = types.StructType([
    types.StructField('review_id', types.StringType()),
    types.StructField('user_id', types.StringType()),
    types.StructField('business_id', types.StringType()),
    types.StructField('stars', types.FloatType()),
    types.StructField('useful', types.IntegerType()),
    types.StructField('funny', types.IntegerType()),
    types.StructField('cool', types.IntegerType()),
    types.StructField('text', types.StringType()),
    types.StructField('date', types.TimestampType()),
    ])

    # Read business ids from business table
    business_ids= spark.read.parquet(business_file).select('business_id')
    # Read reviews json
    all_reviews = spark.read.json(reviews_file, schema=reviews_schema).repartition(16)
    # Keep reviews liked to business ids of interest
    filtered_reviews = all_reviews.join(broadcast(business_ids), on='business_id')
    filtered_reviews = filtered_reviews.select('review_id', 'user_id', 'business_id', 'stars', 'useful', functions.col('text').alias('contents'), 'date')

    filtered_reviews.write.mode("overwrite").parquet(outdir + '/reviews.parquet')


if __name__ == '__main__':
    business_file = sys.argv[1]
    reviews_file = sys.argv[2]
    outdir = sys.argv[3]
    spark = SparkSession.builder.appName('Reviews Parquet').getOrCreate()
    assert spark.version >= '3.0' # make sure we have Spark 3.0+
    spark.sparkContext.setLogLevel('WARN')
    sc = spark.sparkContext
    main(business_file, reviews_file, outdir)
