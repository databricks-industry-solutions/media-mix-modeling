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

def fix_fastprogress():
    """Run this to get progress bars in pymc3 before importing it on DBR 11.3+."""
    import fastprogress as fp
    from fastprogress.fastprogress import force_console_behavior
    fp.master_bar, fp.progress_bar = force_console_behavior()
    fp.fastprogress.master_bar, fp.fastprogress.progress_bar = force_console_behavior()

fix_fastprogress()

# COMMAND ----------

gold_table_name = "fivetran"
print(f"generated gold_table_name: {gold_table_name}")

# COMMAND ----------

import mlflow
mlflow.set_experiment('/Users/{}/media_mix'.format(current_user))

# COMMAND ----------


