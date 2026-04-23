
from pyspark.sql import SparkSession
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.feature import VectorAssembler, StringIndexer
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.sql.functions import lit

if __name__ == "__main__":
    # Initialize Spark Session
    spark = SparkSession.builder.appName("Iris_RandomForest_FullMetrics").getOrCreate()

    # 1. Load Data
    df = spark.read.csv("hdfs:///user/maria_dev/assignment_1/iris.csv", header=True, inferSchema=True)

    # 2. Synchronize Indexing
    indexer = StringIndexer(inputCol="species", outputCol="label").fit(df)
    df_indexed = indexer.transform(df)

    # 3. Assemble Features
    feature_cols = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
    assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
    df_final = assembler.transform(df_indexed)

    # 4. Split Data (Consistent seed=42)
    train_df, test_df = df_final.randomSplit([0.7, 0.3], seed=42)

    # 5. Build Random Forest Model
    rf = RandomForestClassifier(labelCol="label", featuresCol="features")

    # 6. Hyperparameter Grid

    paramGrid = ParamGridBuilder() \
        .addGrid(rf.numTrees, [5, 10, 15]) \
        .addGrid(rf.maxDepth, [5, 7, 9]) \
        .addGrid(rf.minInstancesPerNode, [5]) \
        .addGrid(rf.subsamplingRate, [0.8, 0.9, 1.0]) \
        .build()

    # Base evaluator
    evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction")

    # 7. Cross-Validation (3-Fold)
    cv = CrossValidator(estimator=rf,
                        estimatorParamMaps=paramGrid,
                        evaluator=evaluator,
                        numFolds=3)

    print("\nTuning Random Forest...")
    cvModel = cv.fit(train_df)
    bestModel = cvModel.bestModel

    # 8. Extraction of Tuned Parameters
    print("\n" + "="*40)
    print("OPTIMIZED RANDOM FOREST PARAMETERS")
    print("-" * 40)
    print("Best numTrees: " + str(bestModel._java_obj.getNumTrees()))
    print("Best maxDepth: " + str(bestModel._java_obj.getMaxDepth()))
    print("Best minInstancesPerNode: " + str(bestModel._java_obj.getMinInstancesPerNode()))
    print("Best subsamplingRate: " + str(bestModel._java_obj.getSubsamplingRate()))
    print("="*40)

    # 9. Feature Importance Ranking
    importances = bestModel.featureImportances
    print("\n" + "="*40)
    print("FEATURE IMPORTANCE RANKING")
    print("-" * 40)
    for i, column in enumerate(feature_cols):
        print("{:<15}: {:.4f}".format(column, importances[i]))
    print("="*40)

    # 10. Final Performance Evaluation (On Test Set Only)
    test_predictions = cvModel.transform(test_df)
    
    accuracy  = evaluator.setMetricName("accuracy").evaluate(test_predictions)
    f1        = evaluator.setMetricName("f1").evaluate(test_predictions)
    precision = evaluator.setMetricName("weightedPrecision").evaluate(test_predictions)
    recall    = evaluator.setMetricName("weightedRecall").evaluate(test_predictions)
    
    print("\n" + "="*40)
    print("RANDOM FOREST TEST METRICS")
    print("-" * 40)
    print("Accuracy:  {:.4f}".format(accuracy))
    print("F1-Score:  {:.4f}".format(f1))
    print("Precision: {:.4f}".format(precision))
    print("Recall:    {:.4f}".format(recall))
    print("="*40 + "\n")

    # 11. Prepare Full Output (Train + Test)
    train_results = cvModel.transform(train_df).withColumn("dataset_type", lit("TRAIN"))
    test_results = test_predictions.withColumn("dataset_type", lit("TEST"))
    
    final_output = train_results.union(test_results)
    
    # Columns to display and save
    display_cols = feature_cols + ["label", "prediction", "probability", "dataset_type"]
    
    print("INSPECTING TUNED RESULTS (Sample):")
    final_output.select(*display_cols).show(20)

    # 12. Save Results for Jupyter
    output_path = "hdfs:///user/maria_dev/assignment_1/rf_results_5"
    final_output.select(*display_cols).write.mode("overwrite").parquet(output_path)

    print("SUCCESS: Results saved to " + output_path)
    spark.stop()
