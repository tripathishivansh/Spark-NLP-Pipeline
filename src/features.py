from pyspark.ml.feature import HashingTF, IDF, Word2Vec

def create_tfidf_features(df):
    """Generate TF-IDF feature vectors."""
    hashingTF = HashingTF(inputCol="filtered_tokens", outputCol="rawFeatures", numFeatures=10000)
    featurizedData = hashingTF.transform(df)
    idf = IDF(inputCol="rawFeatures", outputCol="tfidf_features")
    idf_model = idf.fit(featurizedData)
    tfidf_df = idf_model.transform(featurizedData)
    print("✅ TF-IDF features created.")
    return tfidf_df

def create_word2vec_features(df):
    """Generate Word2Vec embeddings."""
    w2v = Word2Vec(vectorSize=100, minCount=1, inputCol="filtered_tokens", outputCol="word2vec_features")
    model = w2v.fit(df)
    w2v_df = model.transform(df)
    print("✅ Word2Vec features created.")
    return w2v_df
