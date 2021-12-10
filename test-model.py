from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml import PipelineModel
from pyspark.sql import SparkSession, functions, types
import sys
assert sys.version_info >= (3, 5)  # make sure we have Python 3.5+

spark = SparkSession.builder.appName('tmax model tester').getOrCreate()
assert spark.version >= '2.3'  # make sure we have Spark 2.3+
spark.sparkContext.setLogLevel('WARN')


def test_model(model_file, testSet):
    # get the data
    test_set = spark.read.parquet(testSet)

    # load the model
    model = PipelineModel.load(model_file)

    # use the model to make predictions
    predictions = model.transform(test_set).cache()
    # predictions.show()

    # evaluate the predictions
    r2_evaluator = RegressionEvaluator(predictionCol='prediction', labelCol='stars',
                                       metricName='r2')
    r2 = r2_evaluator.evaluate(predictions)

    rmse_evaluator = RegressionEvaluator(predictionCol='prediction', labelCol='stars',
                                         metricName='rmse')
    rmse = rmse_evaluator.evaluate(predictions)

    print('r2 =', r2)
    print('rmse =', rmse)
    predictions.select('features', 'stars', 'prediction').show()
    # If you used a regressor that gives .featureImportances, maybe have a look...
    print(model.stages[-1].featureImportances)


if __name__ == '__main__':
    model_file = sys.argv[1]
    testSet = sys.argv[2]
    test_model(model_file, testSet)
