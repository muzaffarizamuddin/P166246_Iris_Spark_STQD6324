
from pyspark.sql import SparkSession
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.feature import VectorAssembler, StringIndexer
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

if __name__ == "__main__":
    spark = SparkSession.builder.appName("Iris_DecisionTree_Final").getOrCreate()

    # 1. Load Data
    df = spark.read.csv("hdfs:///user/maria_dev/assignment_1/iris.csv", header=True, inferSchema=True)

    # 2. Synchronize Indexing
    indexer = StringIndexer(inputCol="species", outputCol="label").fit(df)
    df_indexed = indexer.transform(df)

    # 3. Assemble Features
    # Decision Trees are scale-invariant, so raw features are fine.
    assembler = VectorAssembler(
        inputCols=["sepal_length", "sepal_width", "petal_length", "petal_width"], 
        outputCol="features")
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

    # 9. Comprehensive Performance Evaluation
    predictions = cvModel.transform(test_df)
    
    # Calculate different metrics by updating the evaluator's metricName
    accuracy = evaluator.setMetricName("accuracy").evaluate(predictions)
    f1 = evaluator.setMetricName("f1").evaluate(predictions)
    precision = evaluator.setMetricName("weightedPrecision").evaluate(predictions)
    recall = evaluator.setMetricName("weightedRecall").evaluate(predictions)
    
    print("\n" + "="*40)
    print("DECISION TREE TEST METRICS")
    print("-" * 40)
    print("Accuracy:  {:.4f}".format(accuracy))
    print("F1-Score:  {:.4f}".format(f1))
    print("Precision: {:.4f}".format(precision))
    print("Recall:    {:.4f}".format(recall))
    print("="*40 + "\n")

    # 10. Save Results for Jupyter
    output_path = "hdfs:///user/maria_dev/assignment_1/dtree_result_2"
    predictions.select("label", "prediction", "probability").write.mode("overwrite").parquet(output_path)

    print("SUCCESS: Results saved to " + output_path)
    spark.stop()
