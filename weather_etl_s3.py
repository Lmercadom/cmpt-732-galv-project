import sys

from pyspark.sql import SparkSession, functions, types
spark = SparkSession.builder.appName('weather ETL') \
    .config('fs.s3a.access.key', 'KEY') \
    .config('fs.s3a.secret.key', 'SECRET') \
    .config('spark.hadoop.fs.s3a.fast.upload.buffer', 'bytebuffer') \
    .config('fs.s3a.endpoint', 'http://s3-us-west-2.amazonaws.com').getOrCreate()


observation_schema = types.StructType([
    types.StructField('station', types.StringType(), False),
    types.StructField('date', types.StringType(), False),
    types.StructField('observation', types.StringType(), False),
    types.StructField('value', types.IntegerType(), False),
    types.StructField('mflag', types.StringType(), False),
    types.StructField('qflag', types.StringType(), False),
    types.StructField('sflag', types.StringType(), False),
    types.StructField('obstime', types.StringType(), False),
])


def main(inputs, output):
    weather = spark.read.csv(inputs, schema=observation_schema).cache()
    weather.show()

    weather = weather.filter(weather['station'].startswith(functions.lit('CA')))
    # or: weather = weather.filter(functions.substring(weather['station'], 0, 2) == functions.lit('CA'))

    weather = weather.filter(weather['qflag'].isNull())
    weather = weather.filter(weather['observation'] == functions.lit('TMAX'))
    weather = weather.select(
        weather['station'],
        weather['date'],
        (weather['value']/10).alias('tmax'),
    )

    # weather.write.json(output, compression='gzip', mode='overwrite')


if __name__ == '__main__':
    inputs = 's3a://galv/test'
    output = 's3a://ggbaker-bigdata/output'
    main(inputs, output)
