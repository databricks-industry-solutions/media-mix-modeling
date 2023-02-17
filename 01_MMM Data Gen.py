# Databricks notebook source
# MAGIC %run ./config/config $reset_all_data=true

# COMMAND ----------

# MAGIC %md ### Data Gen

# COMMAND ----------

from pyspark.sql.functions import date_add, to_date, rand
from pyspark.sql.types import StructType, StructField, DateType, DoubleType

# COMMAND ----------

# define the schema for the data
schema = StructType([
    StructField("date", DateType(), nullable=False),
    StructField("adwords", DoubleType(), nullable=False),
    StructField("facebook", DoubleType(), nullable=False),
    StructField("linkedin", DoubleType(), nullable=False),
    StructField("sales", DoubleType(), nullable=False)
])

# COMMAND ----------

# MAGIC %md TBC
