# from sparkdl.xgboost import XgboostRegressor
# from sparkdl.xgboost import XgboostRegressor
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.regression import GBTRegressor, RandomForestRegressor, FMRegressor, LinearRegression
from pyspark.sql import SparkSession, functions, types
import sys
from pyspark.ml.feature import StringIndexer, VectorAssembler, SQLTransformer
from pyspark.ml import Pipeline
assert sys.version_info >= (3, 5)  # make sure we have Python 3.5+
# add more functions as necessary


def generate_case_statement(columnName, yesList):
    return f"""
              CASE
                WHEN {columnName} in ("None", "none", "u'none'") or {columnName} is null THEN 'UNKNOWN'
                WHEN {columnName} in ("{'","'.join(yesList)}") THEN 'YES'
                ELSE 'NO'
              END
        """


def evaluate(predictions):
    for metric in ['r2', 'rmse']:
        evaluator = RegressionEvaluator(
            predictionCol='prediction',
            labelCol='stars',
            metricName=metric
        )
        score = evaluator.evaluate(predictions)
        print(f'[{metric}] Validation score for the model:', score)


def main(inputs, output):
    # main logic starts here
    restaurants, attributes = inputs.split(',')

    df_attributes = spark.read.parquet(attributes)
    df_rest = spark.read.parquet(restaurants).select('business_id', 'stars')

    df_joined = df_rest.join(df_attributes, 'business_id')
    df_joined.createOrReplaceTempView('JOINED')
    # df.printSchema()
    # return

    df = spark.sql(f"""
          SELECT
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
          stars
          FROM JOINED
          """)

    train, validation, test = df.randomSplit([0.60, 0.20, 0.20])
    train = train.cache()
    validation = validation.cache()
    test = test.cache()

    feature_columns = ["WIFI", "WheelchairAccessible", "Smoking", "RestaurantsTakeOut", "RestaurantsDelivery", "OutdoorSeating",
                       "HasTV", "HappyHour", "DogsAllowed", "BusinessAcceptsCreditCards", "BikeParking", "Alcohol", "BYOB", "BYOBCorkage"]

    indexers = [StringIndexer(inputCol=column, outputCol=column+"_index")
                for column in feature_columns]
    assembler = VectorAssembler(
        inputCols=[column+"_index" for column in feature_columns],
        outputCol='features')
    regressor = GBTRegressor(featuresCol='features',
                             labelCol='stars', maxIter=100, maxDepth=5)

    pipeline = Pipeline(stages=[*indexers, assembler, regressor])
    model = pipeline.fit(train)
    predictions = model.transform(validation).cache()

    predictions.select('features', 'prediction', 'stars').show()

    evaluate(predictions)

    model.write().overwrite().save(output+'/model')
    test.write.parquet(output+'/test-set')
    validation.write.parquet(output+'/validation-set')
    train.write.parquet(output+'/train-set')

    # TESTING
    predictions = model.transform(test)
    evaluate(predictions)
    print(model.stages[-1].featureImportances)


if __name__ == '__main__':
    inputs = sys.argv[1]
    output = sys.argv[2]
    spark = SparkSession.builder.appName('example code').getOrCreate()
    assert spark.version >= '3.0'  # make sure we have Spark 3.0+
    spark.sparkContext.setLogLevel('WARN')
    sc = spark.sparkContext
    main(inputs, output)
