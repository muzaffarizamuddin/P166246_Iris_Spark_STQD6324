# README: Iris Species Classification using Spark MLlib

## Overview
This project demonstrates an end-to-end Machine Learning workflow for classifying Iris flower species using Spark MLlib, executed on a Hadoop cluster (HDP 2.6.5) and optimized for Google Colab. The primary goal is to leverage distributed computing patterns for training, tuning, and evaluating multiple classification models on the classic Iris dataset. The notebook details the entire process from data loading and preprocessing to model training, hyperparameter tuning, evaluation, and finally, local deployment of the best-performing models.

## Project Setup and Execution

### Running on Hadoop Cluster (Original Environment)
The initial development and hyperparameter tuning were performed on a Hortonworks Data Platform (HDP) 2.6.5 cluster. The `iris.csv` dataset is first uploaded to HDFS, and the Python Spark MLlib scripts are executed via `spark-submit` commands on Unix through Putty. The results are then saved back to HDFS and later extracted for analysis.

**Example `spark-submit` command:**
```bash
spark-submit --master yarn logreg_iris.py
```

### Running Locally on Google Colab (Reproducibility)
This Colab notebook offers a fully reproducible environment:
1.  **Data Loading:** The `iris.csv` data is loaded from a UCI ML archive URL and saved locally.
2.  **Model Training (scikit-learn):** The optimized hyperparameters found during the Hadoop cluster runs are applied to train local scikit-learn models. This allows for quick local predictions without needing a Spark environment.
3.  **Model Training (PySpark on Colab):** For full Spark compatibility within Colab, a local Spark session is initialized, and PySpark models are trained using the optimized parameters.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/muzaffarizamuddin/P166246_Iris_Spark_STQD6324/blob/main/P166246_Iris_Spark.ipynb)

## Dataset: Iris Species

The **Iris dataset**, introduced by Ronald Fisher in 1936, is a classic multivariate dataset comprising 150 samples of iris flowers. Each sample belongs to one of three species (Iris setosa, Iris versicolor, or Iris virginica) and has four measured features: sepal length, sepal width, petal length, and petal width. It is a benchmark for classification algorithms due to its well-understood characteristics and varying degrees of class separability. The flowers are difficult to distinguish by eye as you see in below picture:
<img width="635" height="313" alt="IRIS-flower" src="https://github.com/user-attachments/assets/7e5e4e01-3a98-4473-9924-f3b8efa97e06" />

## Methodology

*   **Train/Test Split:** A 70:30 ratio was used for training and testing, respectively.
*   **Models Evaluated:** Logistic Regression, Decision Tree, Random Forest, Naive Bayes, and Multilayer Perceptron (Neural Network).
*   **Optimization:** Each model underwent hyperparameter tuning using Spark's `ParamGridBuilder` and `CrossValidator` with 3-fold cross-validation, optimizing for the F1-Score.
*   **Feature Scaling:** `StandardScaler` was applied to continuous features to prevent scale-based biases in distance-sensitive models.

## Model Comparison and Findings

| Rank | Model | Accuracy | F1-Score | Status |
| :--- | :--- | :---: | :---: | :--- |
| **1** | **Logistic Regression** | **0.9722** | **0.9733** | **Leader** |
| **2** | **Random Forest** | 0.9444 | 0.9466 | Top Tier |
| **3** | **Neural Network (MLP)**| 0.9444 | 0.9466 | Top Tier |
| **4** | **Decision Tree** | 0.9167 | 0.9200 | Mid Tier |
| **5** | **Naive Bayes** | 0.6667 | 0.6032 | Unsuccessful |

**Key Observations:**
*   **Logistic Regression** performed exceptionally well, indicating the dataset's strong linear separability.
*   **Random Forest** and **MLP** also achieved high accuracy, demonstrating the robustness of ensemble and neural network approaches.
*   **Naive Bayes** struggled significantly due to its assumption of feature independence, which is violated by the highly correlated features in the Iris dataset.

**Recommendation:** Logistic Regression is recommended for its superior accuracy, computational efficiency, and interpretability for this dataset.

## Code Structure

The project consists of the main Colab notebook (`P166246_Iris_Spark.ipynb`) and several Python scripts designed for execution on a Hadoop cluster.

### Notebook Sections:
*   **Data Loading and Preprocessing:** Initial loading of `iris.csv` and preparation for Spark MLlib.
*   **Methodology:** Details on test/train split, model architecture, and optimization strategies.
*   **Model-Specific Sections (2.1 - 2.5):** Each section analyzes one of the five models, presenting hyperparameter tuning details, performance metrics, confusion matrices, and ROC curves.
*   **Model Comparison:** A summary table and visualization comparing the performance of all models.
*   **Local Prediction:** Demonstrations of prediction using scikit-learn and PySpark within the Colab environment.

### Python Scripts (for Hadoop/Spark Cluster):
*   `logreg_iris.py`: Spark MLlib pipeline for Logistic Regression.
*   `dtree_iris.py`: Spark MLlib pipeline for Decision Tree Classifier.
*   `rf_iris.py`: Spark MLlib pipeline for Random Forest Classifier.
*   `nb_iris.py`: Spark MLlib pipeline for Naive Bayes Classifier.
*   `mlp_iris.py`: Spark MLlib pipeline for Multilayer Perceptron Classifier.

Each script follows a similar structure:
1.  **Spark Session Initialization**
2.  **Data Loading and Preparation** (Feature Engineering, Label Indexing)
3.  **Model Definition and Hyperparameter Grid (`ParamGridBuilder`)**
4.  **Cross-Validation (`CrossValidator`)**
5.  **Model Fitting**
6.  **Evaluation**
7.  **Results Saving** (to Parquet files on HDFS)

## Dependencies

*   Apache Spark 3.x (with PySpark)
*   Hadoop Distributed File System (HDFS)
*   Python 3.x
*   pandas
*   numpy
*   matplotlib
*   seaborn
*   scikit-learn
*   requests
*   Pillow (PIL)

## References

See the `## References` section in the notebook for a detailed list of academic sources.
