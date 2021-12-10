import sys
assert sys.version_info >= (3, 5)

from pyspark.sql import SparkSession, functions, types
from pyspark.sql import functions as f
from pyspark.sql.functions import countDistinct
from pyspark.sql.types import IntegerType


def main(inputs,rest_input, output):
    df = spark.read.json(inputs)

    df_filter = df.select ("business_id","latitude","longitude","categories","name") #selecting columns required

    df_restaurants = spark.read.parquet(rest_input)
    df_restaurants.cache()
    #rest_input is input file ingested to filter the business as restaurant.

    df_rest = df_restaurants.withColumnRenamed("business_id", "rest_id").withColumnRenamed("name", "name_rest")\
            .withColumnRenamed("latitude", "latitude_rest").withColumnRenamed("longitude", "longitude_rest")

    df_join = df_filter.join(df_rest, df_rest.rest_id == df_filter.business_id, "inner")
    df_business = df_join.select("business_id","name","longitude","latitude","categories")

    df_final = df_business.withColumn("longitude_int", df_business.longitude.cast(IntegerType()))\
        .withColumn("latitude_int", df_business.latitude.cast(IntegerType()))
    df_final = df_final.repartition(16)

    df_final.write.option("sep", "^").option("header", "true").csv(output)


if __name__ == '__main__':
    inputs = sys.argv[1]
    rest_input = sys.argv[2]
    output = sys.argv[3]
    spark = SparkSession.builder.appName('user_ingestion').getOrCreate()
    assert spark.version >= '3.0'  # make sure we have Spark 3.0+
    spark.sparkContext.setLogLevel('WARN')
    sc = spark.sparkContext
    main(inputs,rest_input,output)
