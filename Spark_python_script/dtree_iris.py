
from pyspark.sql import SparkSession
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.feature import VectorAssembler, StringIndexer
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.sql.functions import lit

if __name__ == "__main__":
    # Initialize Spark Session
    spark = SparkSession.builder.appName("Iris_DecisionTree_Final").getOrCreate()

    # 1. Load Data
    df = spark.read.csv("hdfs:///user/maria_dev/assignment_1/iris.csv", header=True, inferSchema=True)

    # 2. Synchronize Indexing
    indexer = StringIndexer(inputCol="species", outputCol="label").fit(df)
    df_indexed = indexer.transform(df)

    # 3. Assemble Features
    # Decision Trees are scale-invariant, so raw features are fine.
    feature_cols = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
    assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
    df_final = assembler.transform(df_indexed)

    # 4. Split Data (Consistent with other models)
    train_df, test_df = df_final.randomSplit([0.8, 0.2], seed=42)

    # 5. Build Decision Tree Model
    dt = DecisionTreeClassifier(labelCol="label", featuresCol="features")

    # 6. Hyperparameter Grid
    paramGrid = ParamGridBuilder() \
        .addGrid(dt.maxDepth, [2, 5, 10]) \
        .addGrid(dt.impurity, ["gini", "entropy"]) \
        .build()

    # Base evaluator
    evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction")

    # 7. Cross-Validation
    cv = CrossValidator(estimator=dt,
                        estimatorParamMaps=paramGrid,
                        evaluator=evaluator,
                        numFolds=3)

    print("\nTuning Decision Tree...")
    cvModel = cv.fit(train_df)
    bestModel = cvModel.bestModel

    # 8. Extraction of Tuned Parameters
    print("\n" + "="*40)
    print("OPTIMIZED TREE PARAMETERS")
    print("-" * 40)
    print("Best maxDepth: " + str(bestModel._java_obj.getMaxDepth()))
    print("Best impurity: " + str(bestModel._java_obj.getImpurity()))
    print("="*40)

    # 9. Final Performance Evaluation (On Test Set)
    test_predictions = cvModel.transform(test_df)
    
    accuracy = evaluator.setMetricName("accuracy").evaluate(test_predictions)
    f1 = evaluator.setMetricName("f1").evaluate(test_predictions)
    precision = evaluator.setMetricName("weightedPrecision").evaluate(test_predictions)
    recall = evaluator.setMetricName("weightedRecall").evaluate(test_predictions)
    
    print("\n" + "="*40)
    print("DECISION TREE TEST METRICS")
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
    output_path = "hdfs:///user/maria_dev/assignment_1/dtree_results_3"
    final_output.select(*display_cols).write.mode("overwrite").parquet(output_path)

    print("SUCCESS: Results saved to " + output_path)
    spark.stop()
