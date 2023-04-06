# Databricks notebook source
# MAGIC %md This accelerator notebook is available at https://github.com/databricks-industry-solutions/media-mix-modeling. Please use the `mmm_cluster` cluster created by RUNME notebook at the root directory of this accelerator folder, if you run this notebook interactively.

# COMMAND ----------

# MAGIC %run ./config/config $reset_all_data=false

# COMMAND ----------

# MAGIC %md
# MAGIC **Outline for MMM solution accelerator**
# MAGIC 
# MAGIC * Configure environment
# MAGIC * Prepare data
# MAGIC * Build model
# MAGIC   * Model specification, Model fitting, Posterior analysis
# MAGIC * Check model
# MAGIC * Calibrate model using external benchmarks (databricks' historical benchmarks)
# MAGIC * Analyze efficiency using saturation curves
# MAGIC   * Low saturation = room to grow
# MAGIC   * High saturation = could taper back a bit
# MAGIC   * Use this to find optimal spend for a given channel
# MAGIC * Analyze ad stock / decay curves to understand the lag for given channel
# MAGIC   * Long decay means we could spend a little less during a given week and extract extra value and efficiency.. more bang for buck
# MAGIC * Simulate different dependent variables given x   
# MAGIC * Plug parameters into constrained optimization algorithm. Constraints include:
# MAGIC   * Total budget
# MAGIC   * Budget we’re willing to put toward any channel
# MAGIC   * This results in info on how to shift budget around taking into account ad stock and decay… shows what’s efficient (increase spend) and what’s not (reduce spend)

# COMMAND ----------

# MAGIC %md ### Step 1: Set up the environment

# COMMAND ----------

import pymc3 as pm
import arviz as az
import datetime
import numpy as np
import matplotlib.pyplot as plt
import theano
import theano.tensor as tt
import mlflow
import pickle
from pprint import pprint

import mediamix.model as mmm

print(f"Running on PyMC3 v{pm.__version__}")

RANDOM_SEED = 8927
np.random.seed(RANDOM_SEED)
az.style.use('arviz-darkgrid')

%config InlineBackend.figure_format = 'retina'

# COMMAND ----------

from importlib import reload
reload(mmm)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 2: Load the data
# MAGIC 
# MAGIC The generated dataset simulates a gold table where the input table has been
# MAGIC transformed so the ad spend is a window leading up to the sale rather than 
# MAGIC aggregated up on the same day a sale occured.

# COMMAND ----------

## create temp view of dataset and write to deltalake as delta table mmm_layla_yang.mmm_dateset
## Then create same mmm_dateset table with two downstream tables with log transform / interaction terms to demostrate Lineage under the catalog.schema "field_demos.uc_lineage"
sdf = spark.table(gold_table_name)
sdf.createOrReplaceTempView("mmm_dataset")

# COMMAND ----------

dbName = 'mmm_layla_yang'
cloud_storage_path = '/Users/layla.yang@databricks.com/field_demos/media/mmm/'
delta_location = '/Users/layla.yang@databricks.com/field_demos/media/mmm/delta/'

spark.sql(f"""create database if not exists {dbName} LOCATION '{cloud_storage_path}/tables' """)
spark.sql(f"""USE {dbName}""")

# COMMAND ----------

# dbutils.fs.rm(delta_location, True)

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE TABLE IF NOT EXISTS mmm_layla_yang.mmm_dateset
# MAGIC USING DELTA
# MAGIC LOCATION '/Users/layla.yang@databricks.com/field_demos/media/mmm/delta/'
# MAGIC AS SELECT *
# MAGIC FROM mmm_dataset

# COMMAND ----------

df = spark.table(gold_table_name).toPandas()
display(df)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 3: Configure the model

# COMMAND ----------

config_path = 'config/model/basic_config.yaml'
config = mmm.ModelConfig.from_config_file(config_path)
pprint(config.to_config_dict())

# COMMAND ----------

# MAGIC %md ### Step 4: Run inference

# COMMAND ----------

params = {
    'draws': 1000,
    'tune': 1000,
    'init': 'auto'}

model, idata, scalers = config.run_inference(params, df)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 5: Analyze the results

# COMMAND ----------

az.summary(idata)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 6: Inspect the trace visually

# COMMAND ----------

az.plot_trace(idata);

# COMMAND ----------

# MAGIC %md 
# MAGIC ### Step 7: Inspect posterior predictive samples

# COMMAND ----------

az.plot_ppc(idata);

# COMMAND ----------

# MAGIC %md 
# MAGIC ### References
# MAGIC [From: Introduction to PyMC3](https://github.com/junpenglao/PrecisionWorkshop1_Prep/blob/master/notebooks/1a.%20Introduction%20to%20PyMC3.ipynb)
# MAGIC * Gelman et al. [3] break down the business of Bayesian analysis into three primary steps:
# MAGIC * Specify a full probability model, including all parameters, data, transformations, missing values and predictions that are of interest.
# MAGIC * Calculate the posterior distribution of the unknown quantities in the model, conditional on the data.
# MAGIC * Perform model checking to evaluate the quality and suitablility of the model.
