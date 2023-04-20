# Databricks notebook source
# MAGIC %md This accelerator notebook is available at https://github.com/databricks-industry-solutions/media-mix-modeling. Please use the `mmm_cluster` cluster created by RUNME notebook at the root directory of this accelerator folder if you run this notebook interactively.

# COMMAND ----------

# MAGIC %md 
# MAGIC ## Media Mix Modeling Accelerator
# MAGIC 
# MAGIC MMM (Marketing or Media Mix Modeling), is a data-driven methodology that enables companies to identify and measure the impact of their marketing campaigns across multiple channels.  MMM helps businesses make better-informed decisions about their advertising and marketing strategies. 
# MAGIC 
# MAGIC By analyzing data from various channels such as TV, social media, email marketing, and more, MMM can determine which channels are contributing the most to strategic KPIs, such as sales. By including external events and indicators, decision makers can better understand the impact of outside factors (such as holidays, economic conditions, or weather) and avoid accidently over-valuing the impact of ad spend alone.
# MAGIC 
# MAGIC Databricks Lakehouse offers a unified platform for building modernized MMM solutions that are scalable and flexible. Marketers can unifies various upstream data sources, automates data ingestion, processing, and transformation, and provides full transparency and traceability of data usage. With powerful DSML capabilities and collaborative workstream, marketers can seamlessly leverage the full potential of their data, driving more informed and effective marketing investment decisions. 
# MAGIC 
# MAGIC In this notebook, we set up the environment to generate some synthetic data we will use in the second notebook, where we showcase how to build a media mix model using the classes we wrote based on pymc3.
# MAGIC 
# MAGIC <insert architecture diagram>

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
