# Databricks notebook source
# MAGIC %run ./config/config $reset_all_data=false

# COMMAND ----------

# MAGIC %md ### Data Gen

# COMMAND ----------

import pyspark.sql
from pyspark.sql import functions as F
from pyspark.sql.types import StructType, StructField, DateType, DoubleType

import numpy as np
import pymc3 as pm
import pandas as pd
import arviz as az

from pprint import pprint

from mediamix.transforms import geometric_adstock_tt, logistic_function
from mediamix.datasets import load_original_fivetran
from mediamix import generator as mmg

from warnings import filterwarnings
filterwarnings('ignore', 'iteritems is deprecated')

%config InlineBackend.figure_format = 'retina'

RANDOM_SEED = 8927
np.random.seed(RANDOM_SEED)
az.style.use('arviz-darkgrid')

# COMMAND ----------

# define the schema for the data
schema = StructType([
    StructField("date", DateType(), nullable=False),
    StructField("adwords", DoubleType(), nullable=False),
    StructField("facebook", DoubleType(), nullable=False),
    StructField("linkedin", DoubleType(), nullable=False),
    StructField("sales", DoubleType(), nullable=False)])

# COMMAND ----------

gen_config_path = 'config/generator/basic_config.yaml'
gen_config = mmg.load_generator_config(gen_config_path)
pprint(gen_config)

# COMMAND ----------

def generate_sales_from_adspend_model(df, config):
    """Generate the sales column from the generated dataframe and config.
    """
    n = len(df)

    # generate some whitenoise for the sales (small, since it will be scaled)
    if True:
        sales = np.random.normal(0, 0.01, size=n)
    else:
        sales = np.zeros(n)

    # add the impact for each channel
    for c in config['channels']:
        sales += mmg.channel_impact(df[c], **config['media'][c])
    
    # return the rescaled generated sales
    return mmg.rescale(sales, config['outcome']['min'], config['outcome']['max'])


def generate_adspend_dataset(config):
    """Generate an adspend dataset based on the given config.
    """
    n = config['n']

    df = mmg.create_empty_dataframe(n, config['channels'])
    for c in config['channels']:
        cc = config['media'][c]
        x = mmg.generate_base_adspend_signal(n)
        x = mmg.rescale(x, cc['min'], cc['max'])
        df[c] = np.round(x, 2)
    df['sales'] = np.round(generate_sales_from_adspend_model(df, config), 2)
    return df


def convert_to_spark_dataframe(df: pd.DataFrame, schema: StructType) -> pyspark.sql.DataFrame:
    """Convert the generator pandas DataFrame to the expected Spark DataFrame.
    """
    return (
        spark.createDataFrame(
            df.reset_index(drop=False)
            .rename({'index': 'date'}), 
            schema=schema))


# COMMAND ----------

df = generate_adspend_dataset(gen_config)
df.loc[:, gen_config['channels'] + ['sales']].plot(linewidth=0.25);

# COMMAND ----------

sdf = convert_to_spark_dataframe(df, schema)
display(sdf)

# COMMAND ----------

sdf.write.mode('overwrite').saveAsTable(output_table_name)
