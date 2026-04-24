
from pyspark.sql import SparkSession
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.feature import VectorAssembler, StringIndexer, StandardScaler
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

def run_optimization():
    spark = SparkSession.builder.appName("LogReg_Step_Optimizer").getOrCreate()
    
    # 1. Load and Preprocess
    df = spark.read.csv("hdfs:///user/maria_dev/assignment_1/iris.csv", header=True, inferSchema=True)
    indexer = StringIndexer(inputCol="species", outputCol="label").fit(df)
    df_indexed = indexer.transform(df)
    
    assembler = VectorAssembler(inputCols=["sepal_length", "sepal_width", "petal_length", "petal_width"], outputCol="raw_features")
    df_assembled = assembler.transform(df_indexed)
    
    scaler = StandardScaler(inputCol="raw_features", outputCol="features", withStd=True, withMean=True)
    df_final = scaler.fit(df_assembled).transform(df_assembled)

    # 2. Split Data (Using your friend's 70:30 recommendation for stability)
    train_df, test_df = df_final.randomSplit([0.7, 0.3], seed=42)
    evaluator = MulticlassClassificationEvaluator(metricName="accuracy")

    # --- PHASE 1: Find Best regParam (Starting at 0.02, decreasing by 0.001) ---
    best_reg = 0.005
    best_acc = 0.0
    current_reg = 0.02
    
    print("PHASE 1: Optimizing regParam...")
    while current_reg >= 0:
        lr = LogisticRegression(regParam=current_reg, elasticNetParam=0.0, maxIter=20)
        model = lr.fit(train_df)
        acc = evaluator.evaluate(model.transform(test_df))
        
        print("Testing regParam: {:.3f} | Accuracy: {:.4f}".format(current_reg, acc))
        
        if acc >= best_acc:
            best_acc = acc
            best_reg = current_reg
        else:
            # Accuracy reduced, stop search
            print("Accuracy dropped. Stopping Phase 1.")
            break
        
        current_reg -= 0.001

    # --- PHASE 2: Find Best elasticNetParam (Starting at 0, increasing by 0.0002) ---
    best_elastic = 0.0
    current_elastic = 0.0
    
    print("\nPHASE 2: Optimizing elasticNetParam using regParam: {:.3f}".format(best_reg))
    while current_elastic <= 1.0:
        lr = LogisticRegression(regParam=best_reg, elasticNetParam=current_elastic, maxIter=20)
        model = lr.fit(train_df)
        acc = evaluator.evaluate(model.transform(test_df))
        
        print("Testing elasticNetParam: {:.4f} | Accuracy: {:.4f}".format(current_elastic, acc))
        
        if acc >= best_acc:
            best_acc = acc
            best_elastic = current_elastic
        else:
            print("Accuracy dropped. Stopping Phase 2.")
            break
            
        current_elastic += 0.00002

    print("\n" + "="*40)
    print("FINAL OPTIMIZED PARAMETERS")
    print("Best regParam: {:.3f}".format(best_reg))
    print("Best elasticNetParam: {:.4f}".format(best_elastic))
    print("Highest Accuracy: {:.4f}".format(best_acc))
    print("="*40)

    spark.stop()

if __name__ == "__main__":
    run_optimization()
