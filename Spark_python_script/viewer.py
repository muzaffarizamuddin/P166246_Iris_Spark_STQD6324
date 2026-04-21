
from pyspark.sql import SparkSession

if __name__ == "__main__":
    spark = SparkSession.builder.getOrCreate()
    # Read the entire folder (easier than typing the long filename)
    path = "hdfs:///user/maria_dev/assignment_1/dtree_result"
    df = spark.read.parquet(path)
    
    print("\n" + "="*50)
    print("INSPECTING TUNED RESULTS")
    print("="*50)
    df.show(50, False)
    print("="*50 + "\n")
    
    spark.stop()
