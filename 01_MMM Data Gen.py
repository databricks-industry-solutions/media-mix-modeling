# Databricks notebook source
# MAGIC %pip install --no-deps typing-extensions==3.7.4 deprecat==2.1.1 semver==2.13.0 cftime==1.6.2 \
# MAGIC     fastprogress==0.2.0 netCDF4==1.6.2 xarray-einstats==0.3 xarray==0.21.0 theano-pymc==1.1.2 \
# MAGIC     arviz==0.11.0 pymc3==3.11.5

# COMMAND ----------

# MAGIC %run ./config/config $reset_all_data=false

# COMMAND ----------

# MAGIC %md ### Data Gen

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

# define the schema for the data
schema = StructType([
    StructField("date", DateType(), nullable=False),
    StructField("adwords", DoubleType(), nullable=False),
    StructField("facebook", DoubleType(), nullable=False),
    StructField("linkedin", DoubleType(), nullable=False),
    StructField("sales", DoubleType(), nullable=False)])

# COMMAND ----------

config_path = 'config/generator/basic_config.yaml'
generator = mmg.Generator.from_config_file(config_path)
df = generator.sample()
df.plot(linewidth=0.25);

# COMMAND ----------

sdf = mmg.convert_to_spark_dataframe(df, schema)
display(sdf)

# COMMAND ----------

sdf.write.mode('overwrite').saveAsTable(gold_table_name)
