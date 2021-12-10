from pyspark.sql import SparkSession, functions, types
import sys

assert sys.version_info >= (3, 5)  # make sure we have Python 3.5+

# add more functions as necessary
#BUCKET = 'galv'

city_mapping = {'%westm%': 'New Westminster', 'Altamonte Springs%': 'Altamonte Springs', 'Atlanta%': 'Atlanta', 'Austin%': 'Austin', 'Beaverton%': 'Beaverton', 'Bee%': 'Bee Cave', 'Berkshire%': 'Berkshire', 'Boston%': 'Boston', 'Brookline%': 'Brookline Village', 'Champions%': 'Champions Gate', 'Clark%': 'Clarkston', 'College%': 'College Park', 'De bary': 'DeBary', 'Dorchester%': 'Dorchester', 'Grandview%': 'Grandview', 'Grove Port%': 'Groveport', 'Hapevile': 'Hapeville', 'Hiliard': 'Hilliard', 'Holb%': 'Holbrook', 'Jeffries Point%': 'Jeffries Point',
                'Kissim%': 'Kissimmee', 'Lake Buena Visa': 'Lake Buena Vista', 'Marlbehead': 'Marblehead', 'Milwaukee': 'Milwaukie', 'Needham%': 'Needham', 'Newton Cent%': 'Newton Centre', 'N%Vancouver%': 'North Vancouver', 'Orlan%': 'Orlando', 'Port%John': 'Port St. John', 'Portland%': 'Portland', 'Quincy%': 'Quincy', 'Roxbury%': 'Roxbury', 'Saint%Cloud': 'Saint Cloud', 'Sandy Spring%': 'Sandy Springs', 'Sanford%': 'Sanford', 'Sommerville': 'Somerville', 'So%Weymouth': 'South Weymouth', 'St%loud': 'St. Cloud', 'Wellesley%': 'Wellesley', 'Winter%park': 'Winter Park'}


def generate_city_case_statement():
    replacement_string = ""
    for items in city_mapping.keys():
        replacement_string += "WHEN city LIKE '" + \
            items + "' THEN '" + city_mapping[items] + "' "
    return f"""
        CASE
            {replacement_string}
            ELSE initcap(city)
        END
    """


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
        f"""select business_id,
          name,
          address,
          latitude,
          longitude,
          {generate_city_case_statement()} as city,
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

    attributes_df = spark.sql("""
                               select Original.business_id,
                               attributes.*
                               from Original
                               where Original.business_id in (select distinct Restaurants.business_id from Restaurants) 
                               """)
    hours_df = spark.sql("""
                               select Original.business_id,
                               hours.*
                               from Original 
                               where Original.business_id in (select distinct Restaurants.business_id from Restaurants) 
                               """)

    categories_df = categories_df.repartition(16)
    business_df = business_df.repartition(16)
    restaurants_df = restaurants_df.repartition(16)
    attributes_df = attributes_df.repartition(16)
    hours_df = hours_df.repartition(16)

    (businessTable, categoriesTable, restaurantsTable,
     attributesTable, hoursTable) = table.split(',')

    business_df.write.mode("overwrite").parquet(
        f"output/{businessTable}.parquet")
    categories_df.write.mode("overwrite").parquet(
        f"output/{categoriesTable}.parquet")
    restaurants_df.write.mode("overwrite").parquet(
        f"output/{restaurantsTable}.parquet")
    attributes_df.write.mode("overwrite").parquet(
        f"output/{attributesTable}.parquet")
    hours_df.write.mode("overwrite").parquet(
        f"output/{hoursTable}.parquet")

   # utils.upload_files_to_s3('output', BUCKET)


if __name__ == '__main__':
    inputs = sys.argv[1]
    table = sys.argv[2]
    spark = SparkSession.builder.appName('GALV project').getOrCreate()
    assert spark.version >= '3.0'  # make sure we have Spark 3.0+
    spark.sparkContext.setLogLevel('WARN')
    sc = spark.sparkContext
    main(inputs, table)
