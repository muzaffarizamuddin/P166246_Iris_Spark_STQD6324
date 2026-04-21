
from pyspark.sql import SparkSession
from pyspark.ml.classification import NaiveBayes
from pyspark.ml.feature import VectorAssembler, StringIndexer, MinMaxScaler
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

if __name__ == "__main__":
    spark = SparkSession.builder.appName("Iris_NaiveBayes_Final").getOrCreate()

    # 1. Load Data
    df = spark.read.csv("hdfs:///user/maria_dev/assignment_1/iris.csv", header=True, inferSchema=True)

    # 2. Synchronize Indexing
    indexer = StringIndexer(inputCol="species", outputCol="label").fit(df)
    df_indexed = indexer.transform(df)

    # 3. Assemble Features
    assembler = VectorAssembler(
        inputCols=["sepal_length", "sepal_width", "petal_length", "petal_width"], 
        outputCol="raw_features")
    df_assembled = assembler.transform(df_indexed)

    # 4. Scale Features
    # Naive Bayes requires non-negative values; MinMaxScaler (0 to 1) is ideal here.
    scaler = MinMaxScaler(inputCol="raw_features", outputCol="features")
    df_final = scaler.fit(df_assembled).transform(df_assembled)

    # 5. Split Data (Consistent seed=42)
    train_df, test_df = df_final.randomSplit([0.8, 0.2], seed=42)

    # 6. Build Naive Bayes Model
    nb = NaiveBayes(labelCol="label", featuresCol="features")

    # 7. Hyperparameter Grid
    paramGrid = ParamGridBuilder() \
        .addGrid(nb.smoothing, [0.1, 1.0, 5.0]) \
        .build()

    # Base evaluator
    evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction")

    # 8. Cross-Validation
    cv = CrossValidator(estimator=nb,
                        estimatorParamMaps=paramGrid,
                        evaluator=evaluator,
                        numFolds=3)

    print("\nTuning Naive Bayes...")
    cvModel = cv.fit(train_df)
    bestModel = cvModel.bestModel

    # 9. Results Extraction
    print("\n" + "="*40)
    print("OPTIMIZED NAIVE BAYES PARAMETERS")
    print("-" * 40)
    print("Best smoothing: " + str(bestModel._java_obj.getSmoothing()))
    print("="*40)

    # 10. Comprehensive Performance Evaluation
    predictions = cvModel.transform(test_df)
    
    accuracy  = evaluator.setMetricName("accuracy").evaluate(predictions)
    f1        = evaluator.setMetricName("f1").evaluate(predictions)
    precision = evaluator.setMetricName("weightedPrecision").evaluate(predictions)
    recall    = evaluator.setMetricName("weightedRecall").evaluate(predictions)
    
    print("\n" + "="*40)
    print("NAIVE BAYES TEST METRICS")
    print("-" * 40)
    print("Accuracy:  {:.4f}".format(accuracy))
    print("F1-Score:  {:.4f}".format(f1))
    print("Precision: {:.4f}".format(precision))
    print("Recall:    {:.4f}".format(recall))
    print("="*40 + "\n")

    # 11. Save for Jupyter
    output_path = "hdfs:///user/maria_dev/assignment_1/nb_result_2"
    predictions.select("label", "prediction", "probability").write.mode("overwrite").parquet(output_path)

    print("SUCCESS: Results saved to " + output_path)
    spark.stop()
