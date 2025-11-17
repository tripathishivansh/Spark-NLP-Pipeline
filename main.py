from pyspark.sql import SparkSession 
from src.config import * 
from src.preprocess import ingest_data, clean_text 
from src.features import create_tfidf_features, create_word2vec_features 
from src.train_spark import train_spark_model 
from src.export_dl import export_to_numpy


def main():
    # Initialize Spark 
    spark = SparkSession.builder.appName("Local NLP Pipeline").config("spark.driver.memory", "8g").config("spark.sql.shuffle.partitions", "4").getOrCreate()

    print("🚀 Starting NLP data pipeline...")

    # Bronze: Ingest
    raw_df = ingest_data(spark, RAW_PATH, BRONZE_PATH)

    # Silver: Clean & Tokenize
    clean_df = clean_text(raw_df)
    clean_df.write.mode("overwrite").parquet(SILVER_PATH)
    print(f"✅ Clean data saved to {SILVER_PATH}")

    # Gold: TF-IDF Features
    tfidf_df = create_tfidf_features(clean_df)
    tfidf_df.write.mode("overwrite").parquet(GOLD_PATH_TFIDF)
    print(f"✅ TF-IDF features saved to {GOLD_PATH_TFIDF}")

    # Gold: Word2Vec Features
    w2v_df = create_word2vec_features(clean_df)
    w2v_df.write.mode("overwrite").parquet(GOLD_PATH_W2V)
    print(f"✅ Word2Vec features saved to {GOLD_PATH_W2V}")

    # Train Spark ML model
    model = train_spark_model(tfidf_df)
    print("🏁 Model training complete.")

    # Export for Deep Learning
    export_to_numpy(clean_df, DL_EXPORT_PATH)

    print("🎯 Pipeline completed successfully.")

if __name__ == "__main__":
    main()


