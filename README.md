# Spark NLP Pipeline — End-to-End Sentiment Analysis (Bronze → Silver → Gold):		
This repository contains a scalable, production-ready NLP pipeline built using Apache Spark, applying the Medallion Architecture (Bronze → Silver → Gold) for clean, modular data engineering and machine learning.  
The project ingests raw Twitter sentiment data, cleans & preprocesses text, generates TF-IDF and Word2Vec features, trains a Spark ML model, and exports cleaned text for deep learning frameworks such as TensorFlow and PyTorch.  

## Features:  
  * End-to-end Spark NLP pipeline
  * Bronze → Silver → Gold medallion architecture
  * Text cleaning, tokenization, stopword removal
  * TF-IDF and Word2Vec feature engineering
  * Spark ML logistic regression model training
  * Export cleaned data for deep learning (TensorFlow/PyTorch)
  * Fully modular code structure under /src

## Medallion Architecture (Bronze → Silver → Gold):  
### Bronze: 
  * Raw Ingestion
  * Loads original CSV
  * Saves unmodified data as Parquet for reproducibility

### Silver:  
  * Cleaning & Tokenization
  * Lowercasing
  * URL removal
  * HTML removal
  * Regex cleaning
  * Tokenization
  * Stopword removal
  * Result stored as clean token lists

### Gold:  
  * ML-Ready Features
  * TF-IDF feature vectors
  * Word2Vec embeddings
  * Clean text exported for DL models
  
## Installation and Setup  
### Clone the repository:  
	git clone https://github.com/tripathishivansh/Spark-NLP-Pipeline.git
	cd Spark-NLP-Pipeline

## Dependencies:  
### Python Environment:  
### Install them with:  
	pip install -r requirements.txt

### Java, Spark & Hadoop Requirements:  
  ``` java == jdk-17 ```  
  ``` apache-spark == 4.0.1 ```  
  ``` hadoop == 3.5.0 ```  
  ``` winutils == 3.5.0 ```  

### Enviroment Setup:
  ``` set JAVA_HOME "C:\Program Files\Java\jdk-17" ```  
  ``` set SPARK_HOME "C:\spark\spark-4.0.1-bin-hadoop3" ```  
  ``` set HADOOP_HOME "C:\hadoop" ```  
  ``` set PATH "%PATH%;%SPARK_HOME%\bin;%HADOOP_HOME%\bin" ```  

### Windows Setup Note — Missing Native Hadoop Files  
  If you encounter Hadoop or Spark errors on Windows (e.g., “Could not load native-hadoop library for your platform”), it usually means the native Hadoop binaries are missing.  
  To fix this:  
    Download hadoop.dll and place it in:  
		``` C:\hadoop\lib\native ```

## Running the Pipeline:
	python main.py
