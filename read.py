import os
from pyspark.sql import SparkSession, functions, types
from pyspark import SparkConf, SparkContext
import sys
# import utils

assert sys.version_info >= (3, 5)  # make sure we have Python 3.5+
# os.environ['PYSPARK_SUBMIT_ARGS'] = "--packages=com.amazonaws:aws-java-sdk-bundle:1.11.271,org.apache.hadoop:hadoop-aws:3.1.2 pyspark-shell"

# add more functions as necessary


def main():
    # main logic starts here
    # x = utils.read_file_from_s3('galv', 'restaurants.parquet/')
    # df = spark.read.parquet('s3n://galv/restaurants.parquet/')

    # df = spark.read.parquet('s3a://galv/restaurants.parquet')
    df = spark.read.option("delimiter", ",").csv(
        "s3a://galv/test.csv", header=True)
    df.show()


if __name__ == '__main__':
    spark = SparkSession.builder.appName('example code').config('fs.s3a.access.key', 'KEY') \
    .config('fs.s3a.secret.key', 'ACCESS') \
    .config("fs.s3a.endpoint", "us-west-2.amazonaws.com").getOrCreate()

    assert spark.version >= '3.0'  # make sure we have Spark 3.0+
    spark.sparkContext.setLogLevel('WARN')

    # spark._jsc.hadoopConfiguration().set("fs.s3a.access.key", )
    # spark._jsc.hadoopConfiguration().set(
    #     "fs.s3a.secret.key", )
    # spark._jsc.hadoopConfiguration().set(
    #     "fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")
    # spark._jsc.hadoopConfiguration().set("com.amazonaws.services.s3.enableV4", "true")
    # spark._jsc.hadoopConfiguration().set("fs.s3a.aws.credentials.provider",
    #                                      "org.apache.hadoop.fs.s3a.BasicAWSCredentialsProvider")
    # spark._jsc.hadoopConfiguration().set("fs.s3a.endpoint", "us-west-2.amazonaws.com")

    sc = spark.sparkContext
    main()
