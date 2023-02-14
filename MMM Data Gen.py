# Databricks notebook source
dbutils.widgets.dropdown("reset_all_data", "false", ["true", "false"], "Reset all data")

# COMMAND ----------

spark.conf.set("spark.databricks.io.cache.enabled", "true")

# COMMAND ----------

import re
db_prefix = "cme"
current_user = dbutils.notebook.entry_point.getDbutils().notebook().getContext().tags().apply('user')
current_user_no_at = current_user[:current_user.rfind('@')]
current_user_no_at = re.sub(r'\W+', '_', current_user_no_at)

dbName = db_prefix+"_"+current_user_no_at + "_dev"
cloud_storage_path = f"/Users/{current_user}/field_demos/{db_prefix}/mmm/"
reset_all = dbutils.widgets.get("reset_all_data") == "true"

if reset_all:
  spark.sql(f"DROP DATABASE IF EXISTS {dbName} CASCADE")
  dbutils.fs.rm(cloud_storage_path, True)

spark.sql(f"""create database if not exists {dbName} LOCATION '{cloud_storage_path}/tables' """)
spark.sql(f"""USE {dbName}""")

print("using cloud_storage_path {}".format(cloud_storage_path))
print("using database {arg1} with location at {arg2}{arg3}".format(arg1= dbName,arg2= cloud_storage_path, arg3='tables/'))

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
