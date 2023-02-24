from pyspark.sql import SparkSession
import pandas as pd
from datetime import datetime, timedelta


def load_original_fivetran(table_name):
    """Load the original fivetran dataset."""
    spark = SparkSession.builder.appName('fivetran').getOrCreate()
    df = spark.table(table_name).toPandas()
    df['date'] = pd.to_datetime(df['date'])
    start_date = datetime(2020, 1, 1)
    end_date = datetime(2023, 1, 31)
    date_filter = (df['date'] >= start_date) & (df['date'] < end_date)
    df = df[date_filter].sort_values('date')
    df = df.set_index(df['date'])
    return df
