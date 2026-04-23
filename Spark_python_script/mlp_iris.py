
from pyspark.sql import SparkSession
from pyspark.ml.classification import MultilayerPerceptronClassifier
from pyspark.ml.feature import VectorAssembler, StringIndexer, StandardScaler
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.sql.functions import lit

if __name__ == "__main__":
    # Initialize Spark Session
    spark = SparkSession.builder.appName("Iris_MLP_Final_FullMetrics").getOrCreate()

    # 1. Load Data
    df = spark.read.csv("hdfs:///user/maria_dev/assignment_1/iris.csv", header=True, inferSchema=True)

    # 2. Synchronize Indexing
    indexer = StringIndexer(inputCol="species", outputCol="label").fit(df)
    df_indexed = indexer.transform(df)

    # 3. Feature Assembly & Scaling
    feature_cols = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
    assembler = VectorAssembler(inputCols=feature_cols, outputCol="raw_features")
    df_assembled = assembler.transform(df_indexed)
    
    scaler = StandardScaler(inputCol="raw_features", outputCol="features", withStd=True, withMean=True)
    df_final = scaler.fit(df_assembled).transform(df_assembled)

    # 4. Split Data (Consistent seed=42)
    train_df, test_df = df_final.randomSplit([0.7, 0.3], seed=42)

    # 5. Define Neural Network Architecture
    layers = [4,5,4,3]
    mlp = MultilayerPerceptronClassifier(layers=layers, seed=42)

    # 6. Hyperparameter Grid
    paramGrid = ParamGridBuilder() \
        .addGrid(mlp.layers, [[4, 5, 4, 3], [4, 8, 3]]) \
        .addGrid(mlp.stepSize, [0.05, 0.1]) \
        .addGrid(mlp.maxIter, [200]) \
        .addGrid(mlp.blockSize, [1]) \
        .build()

    evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction")

    # 7. Cross-Validation (3-Fold)
    cv = CrossValidator(estimator=mlp,
                        estimatorParamMaps=paramGrid,
                        evaluator=evaluator,
                        numFolds=3)

    print("\nStarting Training and Tuning Neural Network (MLP)...")
    cvModel = cv.fit(train_df)

    # 8. Extraction of Winning Parameters (FIXED FOR SPARK 2)
    # We find the index of the best performing model and pull params from the grid
    import numpy as np
    best_index = np.argmax(cvModel.avgMetrics)
    best_params = paramGrid[best_index]

    print("\n" + "="*40)
    print("OPTIMIZED NEURAL NETWORK PARAMS")
    print("-" * 40)
    for p, v in best_params.items():
        print("Best {}: {}".format(p.name, v))
    print("Layers Architecture: {}".format(layers))
    print("="*40)

    # 9. Final Performance Evaluation (On Test Set Only)
    test_predictions = cvModel.transform(test_df)
    
    accuracy  = evaluator.setMetricName("accuracy").evaluate(test_predictions)
    f1        = evaluator.setMetricName("f1").evaluate(test_predictions)
    precision = evaluator.setMetricName("weightedPrecision").evaluate(test_predictions)
    recall    = evaluator.setMetricName("weightedRecall").evaluate(test_predictions)
    
    print("\n" + "="*40)
    print("NEURAL NETWORK (MLP) TEST METRICS")
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
    output_path = "hdfs:///user/maria_dev/assignment_1/mlp_results_5"
    final_output.select(*display_cols).write.mode("overwrite").parquet(output_path)

    print("SUCCESS: Results saved to " + output_path)
    spark.stop()
