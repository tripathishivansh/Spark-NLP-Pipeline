import pandas as pd

def export_to_numpy(df, output_path):
    """Convert tokenized text to plain text and export to Parquet for DL."""
    pandas_df = df.select("filtered_tokens", "label").toPandas()
    pandas_df["text"] = pandas_df["filtered_tokens"].apply(lambda x: " ".join(x) if isinstance(x, list) else "")
    pandas_df.to_parquet(output_path, index=False)
    print(f"✅ Exported clean data for DL: {output_path}")
