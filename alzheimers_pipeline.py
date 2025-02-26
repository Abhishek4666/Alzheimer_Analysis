from pyspark.sql import SparkSession
from pyspark.sql.functions import col, sum
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml import Pipeline
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

def run_pyspark_job():
    spark = SparkSession.builder.appName("EDA").getOrCreate()

    df = spark.read.csv("hdfs://localhost:9000/project/AlzDataset.csv", header=True, inferSchema=True)

    # EDA
    print("Displaying first 5 rows of dataset:")
    df.show(5)

    print("Displaying schema of dataset:")
    df.printSchema()

    print("Displaying summary statistics of dataset:")
    df.describe().show()

    print(f"Total Rows: {df.count()}")
    print(f"Total Columns: {len(df.columns)}")

    # Checking for null values
    print("Displaying number of Null values in each column:")
    df.select([sum(col(c).isNull().cast("int")).alias(c) for c in df.columns]).show()

    # Pre-processing
    categorical_columns = [item[0] for item in df.dtypes if item[1].startswith('string')]
    numerical_columns = [item[0] for item in df.dtypes if item[1].startswith(('double', 'int'))]

    print(f"No of Categorical columns: {len(categorical_columns)}")
    print(categorical_columns)

    print(f"No of Numerical columns: {len(numerical_columns)}")
    print(numerical_columns)

    # Target Column
    target_col = "Alzheimer_Diagnosis"

    # Encoding
    print("Encoding Categorical features")
    indexers = [StringIndexer(inputCol=col, outputCol=col+"_index", handleInvalid="keep") for col in categorical_columns if col != target_col]

    # Encode target column separately
    target_indexer = StringIndexer(inputCol=target_col, outputCol="label", handleInvalid="keep")

    # Feature assembler (excluding target)
    feature_columns = [col+"_index" for col in categorical_columns if col != target_col] + numerical_columns
    assembler = VectorAssembler(inputCols=feature_columns, outputCol="features")

    # Create pipeline (encoding + feature assembling)
    pipeline = Pipeline(stages=indexers + [target_indexer, assembler])

    # Transform dataset
    df = pipeline.fit(df).transform(df)

    # Splitting data (80% train, 20% test)
    train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)

    # Train Decision Tree Classifier
    dt = DecisionTreeClassifier(labelCol="label", featuresCol="features")
    model = dt.fit(train_df)

    # Predictions
    predictions = model.transform(test_df)
    predictions.show(10)

    # Evaluate Model
    evaluator = MulticlassClassificationEvaluator(labelCol="label", metricName="accuracy")
    accuracy = evaluator.evaluate(predictions)

    print(f"Test Accuracy: {accuracy}")

    spark.stop()
