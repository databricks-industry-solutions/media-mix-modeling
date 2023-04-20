# Databricks notebook source
# MAGIC %md This accelerator notebook is available at https://github.com/databricks-industry-solutions/media-mix-modeling. Please use the `mmm_cluster` cluster created by RUNME notebook at the root directory of this accelerator folder if you run this notebook interactively.

# COMMAND ----------

# MAGIC %run ./config/config $reset_all_data=false

# COMMAND ----------

# MAGIC %md ### Step 1: Set up the environment

# COMMAND ----------

import numpy as np
from pyspark.sql.types import StructType, StructField, DateType, DoubleType
from mediamix import generator as mmg

from warnings import filterwarnings
filterwarnings('ignore', 'iteritems is deprecated')

%config InlineBackend.figure_format = 'retina'

RANDOM_SEED = 8927
np.random.seed(RANDOM_SEED)

# COMMAND ----------

# MAGIC %md ### Step 2: Define the schema

# COMMAND ----------

# define the schema for the data
schema = StructType([
    StructField("date", DateType(), nullable=False),
    StructField("adwords", DoubleType(), nullable=False),
    StructField("facebook", DoubleType(), nullable=False),
    StructField("linkedin", DoubleType(), nullable=False),
    StructField("sales", DoubleType(), nullable=False)])

# COMMAND ----------

# MAGIC %md ### Step 3: Generate the dataset

# COMMAND ----------

config_path = os.path.join(CONFIG_DIR, 'generator/basic_config.yaml')
generator = mmg.Generator.from_config_file(config_path)
df = generator.sample()
df.plot(linewidth=0.25);

# COMMAND ----------

# MAGIC %md ### Step 4: Write the simulated gold table

# COMMAND ----------

sdf = mmg.convert_to_spark_dataframe(df, schema)
sdf.write.mode('overwrite').saveAsTable(gold_table_name)
display(sdf)
