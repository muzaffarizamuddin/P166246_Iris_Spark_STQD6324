
from pyspark.sql import SparkSession
from pyspark.ml.classification import MultilayerPerceptronClassifier
from pyspark.ml.feature import VectorAssembler, StringIndexer, StandardScaler
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
import numpy as np

if __name__ == "__main__":
    spark = SparkSession.builder.appName("Iris_MLP_Final_FullMetrics").getOrCreate()

    # 1. Load Data
    df = spark.read.csv("hdfs:///user/maria_dev/assignment_1/iris.csv", header=True, inferSchema=True)

    # 2. Synchronize Indexing
    indexer = StringIndexer(inputCol="species", outputCol="label").fit(df)
    df_indexed = indexer.transform(df)

    # 3. Feature Assembly & Scaling
    # Standardizing is critical for Neural Networks to converge properly.
    assembler = VectorAssembler(
        inputCols=["sepal_length", "sepal_width", "petal_length", "petal_width"], 
        outputCol="raw_features")
    df_assembled = assembler.transform(df_indexed)
    
    scaler = StandardScaler(inputCol="raw_features", outputCol="features", withStd=True, withMean=True)
    df_final = scaler.fit(df_assembled).transform(df_assembled)

    # 4. Split Data (Consistent seed=42)
    train_df, test_df = df_final.randomSplit([0.8, 0.2], seed=42)

    # 5. Define Neural Network Architecture
    # 4 inputs -> 2 Hidden Layers (5, 4 nodes) -> 3 outputs
    layers = [4, 5, 4, 3]

    # 6. Build MLP and ParamGrid
    mlp = MultilayerPerceptronClassifier(layers=layers, seed=42)

    paramGrid = ParamGridBuilder() \
        .addGrid(mlp.stepSize, [0.01, 0.1]) \
        .addGrid(mlp.maxIter, [100, 200]) \
        .build()

    # Base evaluator for tuning
    evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction")

    # 7. Cross-Validation
    cv = CrossValidator(estimator=mlp,
                        estimatorParamMaps=paramGrid,
                        evaluator=evaluator,
                        numFolds=3)

    print("\nTraining and Tuning Neural Network (MLP)...")
    cvModel = cv.fit(train_df)
    bestModel = cvModel.bestModel

    # 8. Extraction of Winning Parameters (Spark 2.x Universal Method)
    print("\n" + "="*40)
    print("OPTIMIZED NEURAL NETWORK PARAMS")
    print("-" * 40)
    
    best_index = np.argmax(cvModel.avgMetrics)
    best_params = cvModel.getEstimatorParamMaps()[best_index]

    for p, v in best_params.items():
        print("Best {}: {}".format(p.name, v))

    print("Layers Architecture: {}".format(layers))
    print("="*40)

    # 9. Comprehensive Performance Evaluation
    predictions = cvModel.transform(test_df)
    
    accuracy  = evaluator.setMetricName("accuracy").evaluate(predictions)
    f1        = evaluator.setMetricName("f1").evaluate(predictions)
    precision = evaluator.setMetricName("weightedPrecision").evaluate(predictions)
    recall    = evaluator.setMetricName("weightedRecall").evaluate(predictions)
    
    print("\n" + "="*40)
    print("NEURAL NETWORK (MLP) TEST METRICS")
    print("-" * 40)
    print("Accuracy:  {:.4f}".format(accuracy))
    print("F1-Score:  {:.4f}".format(f1))
    print("Precision: {:.4f}".format(precision))
    print("Recall:    {:.4f}".format(recall))
    print("="*40 + "\n")

    # 10. Save Results for Jupyter
    output_path = "hdfs:///user/maria_dev/assignment_1/mlp_result_2"
    predictions.select("label", "prediction", "probability").write.mode("overwrite").parquet(output_path)

    print("SUCCESS: Results saved to " + output_path)
    spark.stop()
