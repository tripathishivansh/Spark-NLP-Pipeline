from pyspark.sql import functions as F
from pyspark.ml.feature import Tokenizer, StopWordsRemover

def ingest_data(spark, raw_path, bronze_path):
    """Read raw CSV and store as Parquet (Bronze layer)."""
    df = spark.read.csv(raw_path, header=True, inferSchema=True)
    df.write.mode("overwrite").parquet(bronze_path)
    print(f"✅ Ingested raw data and saved to {bronze_path}")
    return df

def clean_text(df):
    """Perform text cleaning, tokenization, and stopword removal."""
    df = df.withColumn("clean_text", F.lower(F.col("tweet")))
    df = df.withColumn("clean_text", F.regexp_replace("clean_text", r'https?://\S+|www\.\S+', ''))
    df = df.withColumn("clean_text", F.regexp_replace("clean_text", r'<.*?>', ''))
    df = df.withColumn("clean_text", F.regexp_replace("clean_text", r'[^a-zA-Z0-9\s]', ' '))
    df = df.withColumn("clean_text", F.regexp_replace("clean_text", r'\s+', ' '))

    tokenizer = Tokenizer(inputCol="clean_text", outputCol="tokens")
    df = tokenizer.transform(df)

    remover = StopWordsRemover(inputCol="tokens", outputCol="filtered_tokens")
    df = remover.transform(df)

    print("✅ Text cleaned and tokenized.")
    return df.select("tweet", "clean_text", "filtered_tokens", "label")
