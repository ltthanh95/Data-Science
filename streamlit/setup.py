import pyspark
import streamlit as st
import pandas as pd
from pyspark.sql import SparkSession
from pyspark import SparkConf

class Session:
    def __init__(self):

        @st.cache_resource
        def create_spark_session():
            try:
                # spark = SparkSession.builder \
                #     .master("local[*]") \
                #     .appName("Climate Change Prediction") \
                #     .config("spark.driver.extraJavaOptions", "-Dio.netty.tryReflectionSetAccessible=true") \
                #     .config("spark.executor.extraJavaOptions", "-Dio.netty.tryReflectionSetAccessible=true") \
                #     .getOrCreate()
                conf = SparkConf().setAppName("lecture-lyon2").setMaster("local")
                spark = SparkSession.builder.config(conf=conf).getOrCreate()
                return spark
            except Exception as e:
                st.error(f"Error initializing Spark: {e}")
                return None
        
        self.spark = create_spark_session()
    
    def read_file(self,uploaded_file):
        if self.spark is None:
            st.error("Spark session not initialized.")
            return None
        if self.spark.sparkContext._jsc.sc().isStopped():
            st.error("Spark context is stopped.")
            return None
        try:
            
            df = self.spark.read \
                .option("delimiter", ",") \
                .option("escape", '"') \
                .option("header", True) \
                .option("encoding", "ISO-8859-1") \
                .csv(uploaded_file, inferSchema=True)
            return df
        except Exception as e:
            st.error(f"Error reading file with Spark: {e}")
            return None

        
        