from pyspark.sql import SparkSession, functions, types
import sys
import utils

assert sys.version_info >= (3, 5)  # make sure we have Python 3.5+

# add more functions as necessary
BUCKET = 'galv'


def main(inputs, table):
    # main logic starts here
    df = spark.read.json(inputs).cache()
    df.createOrReplaceTempView('Original')
    categories_df = df.select('business_id', 'categories')
    categories_df.withColumn('col4', functions.explode(
        functions.split('categories', ','))).createOrReplaceTempView('TEMP')
    categories_df = spark.sql(
        "select business_id, trim(col4) as categories from TEMP")

    business_df = spark.sql(
        """select business_id,
          name,
          address,
          latitude,
          longitude,
          city,
          state,
          stars,
          review_count,
          is_open
        from Original""").cache()
    business_df.createOrReplaceTempView('Business')
    categories_df.createOrReplaceTempView('Categories')
    restaurants_df = spark.sql("""
                               select Business.business_id,
                                name,
                                address,
                                latitude,
                                longitude,
                                city,
                                state,
                                stars,
                                review_count,
                                is_open
                               from Business 
                               join Categories 
                               on Categories.business_id = Business.business_id 
                               where Categories.categories like '%Restaurants' or   
                                    Categories.categories like '%Cafe%'
                               """)

    categories_df = categories_df.repartition(16)
    business_df = business_df.repartition(16)
    restaurants_df = restaurants_df.repartition(16)

    (businessTable, categoriesTable, restaurantsTable) = table.split(',')

    business_df.write.mode("overwrite").parquet(
        f"output/{businessTable}.parquet")
    categories_df.write.mode("overwrite").parquet(
        f"output/{categoriesTable}.parquet")
    restaurants_df.write.mode("overwrite").parquet(
        f"output/{restaurantsTable}.parquet")

    utils.upload_files('output', BUCKET)


if __name__ == '__main__':
    inputs = sys.argv[1]
    table = sys.argv[2]
    spark = SparkSession.builder.appName('GALV project').getOrCreate()
    assert spark.version >= '3.0'  # make sure we have Spark 3.0+
    spark.sparkContext.setLogLevel('WARN')
    sc = spark.sparkContext
    main(inputs, table)
