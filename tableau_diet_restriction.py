from pyspark.sql import SparkSession, functions, types
import sys
assert sys.version_info >= (3, 5)  # make sure we have Python 3.5+


def main(diet, restaurants, output):

    # main logic starts here
    diet_restrictions = spark.read.json(diet)
    restaurants = spark.read.parquet(restaurants)
    df = restaurants.join(
        diet_restrictions, 'business_id', 'left').select('business_id', functions.when(functions.col('DietaryRestrictions').isNull(), "Unknown").otherwise(functions.col('DietaryRestrictions')).alias('Dietary Restrictions'), 'state', 'city', 'latitude', 'longitude')

    df.coalesce(1).write.csv(output, header='true')


if __name__ == '__main__':
    diet = sys.argv[1]
    restaurants = sys.argv[2]
    output = sys.argv[3]
    spark = SparkSession.builder.appName('Tableau DietaryRestrictions').getOrCreate()
    assert spark.version >= '3.0'  # make sure we have Spark 3.0+
    spark.sparkContext.setLogLevel('WARN')
    sc = spark.sparkContext
    main(diet, restaurants, output)
