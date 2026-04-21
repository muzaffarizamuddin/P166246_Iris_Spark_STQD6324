
from pyspark.sql import SparkSession
from pyspark.ml.feature import StringIndexer

spark = SparkSession.builder.getOrCreate()
df = spark.read.csv("hdfs:///user/maria_dev/assignment_1/iris.csv", header=True, inferSchema=True)
indexer = StringIndexer(inputCol="species", outputCol="label").fit(df)

for index, label in enumerate(indexer.labels):
    print("Label {}: {}".format(float(index), label))

spark.stop()
