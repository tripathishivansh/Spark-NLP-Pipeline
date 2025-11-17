from pyspark.ml.classification import LogisticRegression
from pyspark.ml.feature import StringIndexer
from pyspark.ml import Pipeline

def train_spark_model(df):
    """Train a Logistic Regression model using TF-IDF features."""
    label_indexer = StringIndexer(inputCol="label", outputCol="label_idx")
    lr = LogisticRegression(featuresCol="tfidf_features", labelCol="label_idx")
    pipeline = Pipeline(stages=[label_indexer, lr])
    model = pipeline.fit(df)
    print("✅ Spark MLlib model trained successfully.")
    return model
