
from pyspark.sql import SparkSession
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.feature import VectorAssembler, StringIndexer, StandardScaler
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

if __name__ == "__main__":
    # Initialize Spark Session
    spark = SparkSession.builder.appName("Iris_Final_Tuned_Report").getOrCreate()

    # 1. Load Data
    # Ensure iris.csv is in the specified HDFS path
    df = spark.read.csv("hdfs:///user/maria_dev/assignment_1/iris.csv", header=True, inferSchema=True)

    # 2. Synchronize Indexing
    # We fit on the whole dataset to ensure 0.0, 1.0, and 2.0 always refer to the same species
    indexer = StringIndexer(inputCol="species", outputCol="label").fit(df)
    df_indexed = indexer.transform(df)

    # 3. Feature Assembly
    assembler = VectorAssembler(
        inputCols=["sepal_length", "sepal_width", "petal_length", "petal_width"], 
        outputCol="raw_features")
    df_assembled = assembler.transform(df_indexed)

    # 4. Feature Scaling
    # Standardizing features improves the convergence of Logistic Regression
    scaler = StandardScaler(inputCol="raw_features", outputCol="features", withStd=True, withMean=True)
    df_final = scaler.fit(df_assembled).transform(df_assembled)

    # 5. Split Data (80% Train, 20% Test)
    train_df, test_df = df_final.randomSplit([0.8, 0.2], seed=42)

    # 6. Model & Hyperparameter Grid
    lr = LogisticRegression(maxIter=20)
    
    # Grid Search explores combinations of Regularization and ElasticNet mixing
    paramGrid = ParamGridBuilder() \
        .addGrid(lr.regParam, [0.01, 0.1, 0.5]) \
        .addGrid(lr.elasticNetParam, [0.0, 0.5, 1.0]) \
        .build()

    # Evaluator for both tuning and final metrics
    evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction")

    # 7. Cross-Validation (3-Fold)
    cv = CrossValidator(estimator=lr,
                        estimatorParamMaps=paramGrid,
                        evaluator=evaluator,
                        numFolds=3)

    print("\nStarting Training and Hyperparameter Tuning...")
    cvModel = cv.fit(train_df)

    # 8. Extraction of Tuned Parameters
    bestModel = cvModel.bestModel
    
    print("\n" + "="*40)
    print("OPTIMIZED HYPERPARAMETERS")
    print("-" * 40)
    # Using the Java object access to retrieve the specific winning params
    print("Best regParam (Regularization): " + str(bestModel._java_obj.getRegParam()))
    print("Best elasticNetParam (L1/L2 Mix): " + str(bestModel._java_obj.getElasticNetParam()))
    print("="*40)

    # 9. Final Performance Evaluation
    predictions = cvModel.transform(test_df)
    
    accuracy = evaluator.setMetricName("accuracy").evaluate(predictions)
    f1 = evaluator.setMetricName("f1").evaluate(predictions)
    precision = evaluator.setMetricName("weightedPrecision").evaluate(predictions)
    recall = evaluator.setMetricName("weightedRecall").evaluate(predictions)
    
    print("\n" + "="*40)
    print("FINAL TEST METRICS")
    print("-" * 40)
    print("Accuracy:  {:.4f}".format(accuracy))
    print("F1-Score:  {:.4f}".format(f1))
    print("Precision: {:.4f}".format(precision))
    print("Recall:    {:.4f}".format(recall))
    print("="*40 + "\n")

    # 10. Save Results for Jupyter Inspection
    output_path = "hdfs:///user/maria_dev/assignment_1/logreg_results"
    predictions.select("label", "prediction", "probability").write.mode("overwrite").parquet(output_path)

    print("SUCCESS: Results saved to " + output_path)
    spark.stop()
