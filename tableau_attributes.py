from pyspark.sql import SparkSession, functions, types
import sys
assert sys.version_info >= (3, 5)  # make sure we have Python 3.5+


def generate_case_statement(columnName, yesList):
    return f"""
              CASE
                WHEN {columnName} in ("None", "none", "u'none'") or {columnName} is null THEN 'UNKNOWN'
                WHEN {columnName} in ("{'","'.join(yesList)}") THEN 'YES'
                ELSE 'NO'
              END
        """


def main(attributes, restaurants, output):

    # main logic starts here
    attributes = spark.read.parquet(attributes)
    restaurants = spark.read.parquet(restaurants)

    df_joined = restaurants.join(attributes, 'business_id').select('business_id', 'state', 'city', 'latitude', 'longitude', 'WiFi', 'WheelchairAccessible', 'Smoking', 'RestaurantsTakeOut', 'RestaurantsDelivery', 'OutdoorSeating',
                                                                   'HasTV', 'HappyHour', 'DogsAllowed', 'BusinessAcceptsCreditCards', 'BikeParking', 'Alcohol', 'BYOB', 'BYOBCorkage', 'stars', 'RestaurantsPriceRange2')

    df_joined.createOrReplaceTempView('JOINED')

    df_attributes = spark.sql(f"""
          SELECT
          business_id,
          state,
          initcap(city) as city,
          latitude,
          longitude,
          {generate_case_statement('WiFi',["u'paid'","paid","free", "u'free'"])} as WIFI,
          {generate_case_statement('WheelchairAccessible',["True"])} as WheelchairAccessible,
          {generate_case_statement('Smoking',["u'outdoor'","outdoor","yes", "u'yes'"])} as Smoking,
          {generate_case_statement('RestaurantsTakeOut',["True"])} as RestaurantsTakeOut,
          {generate_case_statement('RestaurantsDelivery',["True"])} as RestaurantsDelivery,
          {generate_case_statement('OutdoorSeating',["True"])} as OutdoorSeating,
          {generate_case_statement('HasTV',["True"])} as HasTV,
          {generate_case_statement('HappyHour',["True"])} as HappyHour,
          {generate_case_statement('DogsAllowed',["True"])} as DogsAllowed,
          {generate_case_statement('BusinessAcceptsCreditCards',["True"])} as BusinessAcceptsCreditCards,
          {generate_case_statement('BikeParking',["True"])} as BikeParking,
          {generate_case_statement('Alcohol',["beer_and_wine", "u'beer_and_wine'", "full_bar", "u'full_bar'"])} as Alcohol,
          {generate_case_statement('BYOB',["True"])} as BYOB,
          {generate_case_statement('BYOBCorkage',["yes_corkage","u'yes_corkage'","yes_free","u'yes_free'"])} as BYOBCorkage,
          CASE
            WHEN RestaurantsPriceRange2 ='None' or RestaurantsPriceRange2 is null THEN 'UNKNOWN'
            ELSE RestaurantsPriceRange2
          END as RestaurantsPriceRange,
          stars as rating
          FROM JOINED
          """)

    df_attributes.coalesce(1).write.csv(output, header='true')


if __name__ == '__main__':
    attributes = sys.argv[1]
    restaurants = sys.argv[2]
    output = sys.argv[3]
    spark = SparkSession.builder.appName('Tableau Attributes').getOrCreate()
    assert spark.version >= '3.0'  # make sure we have Spark 3.0+
    spark.sparkContext.setLogLevel('WARN')
    sc = spark.sparkContext
    main(attributes, restaurants, output)
