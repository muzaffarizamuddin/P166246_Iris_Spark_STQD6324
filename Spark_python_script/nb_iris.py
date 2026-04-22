
from pyspark.sql import SparkSession
from pyspark.ml.classification import NaiveBayes
from pyspark.ml.feature import VectorAssembler, StringIndexer, MinMaxScaler
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.sql.functions import lit

if __name__ == "__main__":
    # Initialize Spark Session
    spark = SparkSession.builder.appName("Iris_NaiveBayes_Final").getOrCreate()

    # 1. Load Data
    df = spark.read.csv("hdfs:///user/maria_dev/assignment_1/iris.csv", header=True, inferSchema=True)

    # 2. Synchronize Indexing
    indexer = StringIndexer(inputCol="species", outputCol="label").fit(df)
    df_indexed = indexer.transform(df)

    # 3. Feature Assembly
    feature_cols = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
    assembler = VectorAssembler(inputCols=feature_cols, outputCol="raw_features")
    df_assembled = assembler.transform(df_indexed)

    # 4. Feature Scaling (Naive Bayes requires non-negative features)
    scaler = MinMaxScaler(inputCol="raw_features", outputCol="features")
    df_final = scaler.fit(df_assembled).transform(df_assembled)

    # 5. Split Data (Consistent seed=42)
    train_df, test_df = df_final.randomSplit([0.8, 0.2], seed=42)

    # 6. Model & Hyperparameter Grid
    nb = NaiveBayes(labelCol="label", featuresCol="features")
    paramGrid = ParamGridBuilder() \
        .addGrid(nb.smoothing, [0.1, 1.0, 5.0]) \
        .build()

    evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction")

    # 7. Cross-Validation (3-Fold)
    cv = CrossValidator(estimator=nb,
                        estimatorParamMaps=paramGrid,
                        evaluator=evaluator,
                        numFolds=3)

    print("\nStarting Training and Hyperparameter Tuning (Naive Bayes)...")
    cvModel = cv.fit(train_df)
    bestModel = cvModel.bestModel

    # 8. Extraction of Tuned Parameters
    print("\n" + "="*40)
    print("OPTIMIZED NAIVE BAYES PARAMETERS")
    print("-" * 40)
    print("Best smoothing: " + str(bestModel._java_obj.getSmoothing()))
    print("="*40)

    # 9. Final Performance Evaluation (On Test Set Only)
    test_predictions = cvModel.transform(test_df)
    
    accuracy  = evaluator.setMetricName("accuracy").evaluate(test_predictions)
    f1        = evaluator.setMetricName("f1").evaluate(test_predictions)
    precision = evaluator.setMetricName("weightedPrecision").evaluate(test_predictions)
    recall    = evaluator.setMetricName("weightedRecall").evaluate(test_predictions)
    
    print("\n" + "="*40)
    print("NAIVE BAYES TEST METRICS")
    print("-" * 40)
    print("Accuracy:  {:.4f}".format(accuracy))
    print("F1-Score:  {:.4f}".format(f1))
    print("Precision: {:.4f}".format(precision))
    print("Recall:    {:.4f}".format(recall))
    print("="*40 + "\n")

    # 10. Prepare Full Output (Train + Test)
    train_results = cvModel.transform(train_df).withColumn("dataset_type", lit("TRAIN"))
    test_results = test_predictions.withColumn("dataset_type", lit("TEST"))
    
    final_output = train_results.union(test_results)
    
    # Columns to display and save
    display_cols = feature_cols + ["label", "prediction", "probability", "dataset_type"]
    
    print("INSPECTING TUNED RESULTS (Sample):")
    final_output.select(*display_cols).show(20)

    # 11. Save Results for Jupyter
    output_path = "hdfs:///user/maria_dev/assignment_1/nb_results_3"
    final_output.select(*display_cols).write.mode("overwrite").parquet(output_path)

    print("SUCCESS: Results saved to " + output_path)
    spark.stop()
