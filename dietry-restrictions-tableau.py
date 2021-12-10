import json
from pyspark.sql import SparkSession, functions, types
import sys
assert sys.version_info >= (3, 5)  # make sure we have Python 3.5+


# add more functions as necessary
@functions.udf(returnType=types.ArrayType(types.StringType()))
def explode(row):
    row = row.replace("\'", "\"")
    row = row.replace("False", '\"False\"')
    row = row.replace("True", '\"True\"')
    js = json.loads(row)
    _list = []
    for key in js:
        if js[key] == "True":
            _list.append(key)
    return _list


def main(inputs, output):
    # main logic starts here
    df = spark.read.parquet(inputs).where((functions.col(
        'DietaryRestrictions').isNotNull()) & (functions.col(
            'DietaryRestrictions') != 'None')).select("business_id", "DietaryRestrictions")
    # df.rdd.map()
    df = df.select('business_id', functions.explode(
        explode(df['DietaryRestrictions'])).alias('DietaryRestrictions'))
    df = df.coalesce(1).cache()
    df.write.json(output+"/DietaryRestrictions.json", mode="overwrite")
    df.write.csv(output+"/DietaryRestrictions.csv", mode="overwrite")


if __name__ == '__main__':
    inputs = sys.argv[1]
    output = sys.argv[2]
    spark = SparkSession.builder.appName('example code').getOrCreate()
    assert spark.version >= '3.0'  # make sure we have Spark 3.0+
    spark.sparkContext.setLogLevel('WARN')
    sc = spark.sparkContext
    main(inputs, output)
