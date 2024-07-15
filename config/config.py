# Databricks notebook source
dbutils.widgets.dropdown("reset_all_data", "false", ["true", "false"], "Reset all data")

# COMMAND ----------

import re
import os
import mlflow

db_prefix = "cme"
notebook_context = dbutils.notebook.entry_point.getDbutils().notebook().getContext()
current_user = notebook_context.tags().apply('user')
current_user_no_at = current_user[:current_user.rfind('@')]
current_user_no_at = re.sub(r'\W+', '_', current_user_no_at)

dbName = db_prefix + "_" + current_user_no_at + "_dev"
cloud_storage_path = f"/Users/{current_user}/field_demos/{db_prefix}/mmm/"
reset_all = dbutils.widgets.get("reset_all_data") == "true"

if reset_all:
    spark.sql(f"DROP DATABASE IF EXISTS {dbName} CASCADE")
    dbutils.fs.rm(cloud_storage_path, True)

spark.sql(f"""CREATE DATABASE IF NOT EXISTS {dbName}""")
spark.sql(f"""USE {dbName}""")

print(f"using cloud_storage_path {cloud_storage_path}")
print(f"using database {dbName}")

CODE_DIR = '/Workspace' + os.path.dirname(notebook_context.notebookPath().get())
CONFIG_DIR = os.path.join(CODE_DIR, 'config')
LOCAL_WORKING_DIR = f'/tmp/{current_user}/mmm'
EXPERIMENT_PATH = f'/Users/{current_user}/Media Mix Modeling'
os.makedirs(LOCAL_WORKING_DIR, exist_ok=True)
os.chdir(LOCAL_WORKING_DIR)
mlflow.set_experiment(EXPERIMENT_PATH)

print(f"using code directory {CODE_DIR}")
print(f"using config directory {CONFIG_DIR}")
print(f"using local directory {LOCAL_WORKING_DIR}")
print(f"using experiment path {EXPERIMENT_PATH}")

# COMMAND ----------

gold_table_name = "fivetran"
print(f"generated gold_table_name: {gold_table_name}")
