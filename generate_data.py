# Databricks notebook source
# MAGIC %md This accelerator notebook is available at https://github.com/databricks-industry-solutions/media-mix-modeling. 
# MAGIC
# MAGIC To import this accelerator, please [clone the repo above into your workspace](https://docs.databricks.com/repos/git-operations-with-repos.html) instead of using the `Download .dbc` option. Please run the `RUNME` notebook at the root directory of this accelerator folder to create a cluster and a Workflow. Use the `mmm_cluster` cluster created by the RUNME notebook to run this notebook interactively.

# COMMAND ----------

# MAGIC %md 
# MAGIC ## Media Mix Modeling Accelerator
# MAGIC
# MAGIC MMM (Marketing or Media Mix Modeling), is a data-driven methodology that enables companies to identify and measure the impact of their marketing campaigns across multiple channels.  MMM helps businesses make better-informed decisions about their advertising and marketing strategies. 
# MAGIC
# MAGIC By analyzing data from various channels such as TV, social media, email marketing, and more, MMM can determine which channels are contributing the most to strategic KPIs, such as sales. By including external events and indicators, decision makers can better understand the impact of outside factors (such as holidays, economic conditions, or weather) and avoid accidently over-valuing the impact of ad spend alone.
# MAGIC
# MAGIC Databricks Lakehouse offers a unified platform for building modernized MMM solutions that are scalable and flexible. Marketers can unify various upstream data sources, automates data ingestion, processing, and transformation, and provides full transparency and traceability of data usage. With powerful data science and machine learning capabilities and collaborative workstreams, marketers can seamlessly leverage the full potential of their data, driving more informed and effective marketing investment decisions. 
# MAGIC
# MAGIC In this notebook, we set up the environment to generate some synthetic data we will use in the second notebook, where we showcase how to build a media mix model using the classes we wrote based on [pymc](https://www.pymc.io/welcome.html).
# MAGIC
# MAGIC <insert architecture diagram>

# COMMAND ----------

# MAGIC %run ./config/config $reset_all_data=false

# COMMAND ----------

# MAGIC %md ### Step 1: Set up the environment
# MAGIC
# MAGIC First, we need to import a couple of libraries we use to generate the dataset. Also, we've moved the data generation logic into a separate module so we import that as well. This shows how Databricks Repos and Workspace files works with Notebooks to allow you to easily refactor code into reusable modules as you develop your pipelines and models. We also set a seed to help with reproducibility.

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
# MAGIC
# MAGIC Next, we define a schema for the table we're about to generate. The idea with this accelerator is to simulate a gold table where all the daily aggregate marketing spend has already been cleansed and published into the Lakehouse for our marketing team to use. In your application, a similar pipeline may have already been developed by upstream teams for you, or you can use the Databricks Lakehouse to build your pipeline all the way from ingesting from the raw sources! 
# MAGIC
# MAGIC For your analysis, you may have many more spend channels to analyze, other dependent variables and KPI's to analyze them against, or be working at a higher or lower time series granularity (weekly is common in addition to daily). Also, you may have indicator variables to consider to represent the presence of holidays or other events, or include grouping factors by which you can slice your analysis into multiple models or to include in a larger model, such as brands, business units, or geographical regions. 
# MAGIC
# MAGIC For this example, we'll keep it simple and just generate three marketing spend columns for a couple of example channels, along with a date column and a dependent variable, sales. 

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
# MAGIC
# MAGIC After this we can generate the dataset. The generator code is fairly straightforward, but a little long to include inline in the notebook. It has been refactored into a `Generator` class which contains multiple `Channel` instances, each of which support either a decay effect (geometric adstock), a saturation effect (logistic), or both of those or neither. It also supports a weighting factor called \\(\beta\\) the controls the impact on sales. The specific parameter values are stored in a `yaml` configuration file in the config directory, so feel free to copy that file and experiment with it to more closely mimic your specific business case. You can also explore and extend the `Generator` itself to create your own simulator based on whatever data you'd like to try to recover with your downstream model. We'll say more about the model itself as we get into the modeling notebook later on.

# COMMAND ----------

config_path = os.path.join(CONFIG_DIR, 'generator/basic_config.yaml')
generator = mmg.Generator.from_config_file(config_path)
df = generator.sample()
df.plot(linewidth=0.25);

# COMMAND ----------

# MAGIC %md ### Step 4: Write the simulated gold table
# MAGIC
# MAGIC We generated the dataset above as a *pandas* `DataFrame`, which is fine for a small dataset like this, but in many cases your data will be coming from a gold 🥇 table that is fed by upstream data sources and stored in your Databricks Lakehouse, ideally in [Unity Catalog](https://www.databricks.com/product/unity-catalog). Here, we'll simulate that aspect as well by converting the *pandas* `DataFrame` to a *spark* `DataFrame` and writing it out to a catalog created by our setup script as a [Delta](https://docs.databricks.com/delta/index.html) table.

# COMMAND ----------

sdf = mmg.convert_to_spark_dataframe(df, schema)
sdf.write.mode('overwrite').saveAsTable(gold_table_name)
display(sdf)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Data generation ✅
# MAGIC
# MAGIC That's all for the data generation notebook. Next, head on over to the "02_MMM PyMC Example" notebook to see if we can recover those parameters with our Bayesian modeling exercise!
