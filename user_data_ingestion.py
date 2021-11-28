import sys
assert sys.version_info >= (3, 5) # make sure we have Python 3.5+

from pyspark.sql import SparkSession, functions, types
from pyspark.sql import functions as f

# add more functions as necessary

def main(inputs, output):
    df_user = spark.read.json("/Users/vishakha/Desktop/yelp_dataset/yelp_academic_dataset_user.json")
    #change this to input later passing via comand line
    df = df_user.select("average_stars", "elite", "fans", "name", "review_count","user_id","yelping_since")
    df_elite_status = df.withColumn('elite_status', f.when((df.elite != ''), "yes").otherwise("no"))
    spark.write.json("output")


if __name__ == '__main__':
    inputs = sys.argv[1]
    output = sys.argv[2]
    spark = SparkSession.builder.appName('example code').getOrCreate()
    assert spark.version >= '3.0' # make sure we have Spark 3.0+
    spark.sparkContext.setLogLevel('WARN')
    sc = spark.sparkContext
    main(inputs, output)