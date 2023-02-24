# Databricks notebook source
# MAGIC %run ./config/config $reset_all_data=false

# COMMAND ----------

# MAGIC %md
# MAGIC ### Outline for MMM solution accelerator
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

# MAGIC %md #### Reference
# MAGIC [From: Introduction to PyMC3](https://github.com/junpenglao/PrecisionWorkshop1_Prep/blob/master/notebooks/1a.%20Introduction%20to%20PyMC3.ipynb)
# MAGIC * Gelman et al. [3] break down the business of Bayesian analysis into three primary steps:
# MAGIC * Specify a full probability model, including all parameters, data, transformations, missing values and predictions that are of interest.
# MAGIC * Calculate the posterior distribution of the unknown quantities in the model, conditional on the data.
# MAGIC * Perform model checking to evaluate the quality and suitablility of the model.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Configure environment

# COMMAND ----------

#!pip install pymc3==3.11.5 

# COMMAND ----------

import pymc3 as pm
import arviz as az
import datetime
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import theano
import theano.tensor as tt

from mediamix.transforms import logistic_function, geometric_adstock_tt

# COMMAND ----------

print(f"Running on PyMC3 v{pm.__version__}")

# COMMAND ----------

RANDOM_SEED = 8927
np.random.seed(RANDOM_SEED)
az.style.use('arviz-darkgrid')

# COMMAND ----------

# MAGIC %md
# MAGIC ### Prepare Data

# COMMAND ----------

# MAGIC %md
# MAGIC The generated dataset simulates a gold table where the input table has been
# MAGIC transformed so the ad spend is a window leading up to the sale rather than 
# MAGIC aggregated up on the same day a sale occured.

# COMMAND ----------

df = spark.table(gold_table_name).toPandas()

# COMMAND ----------

from sklearn.preprocessing import MinMaxScaler
# scale variables to be between 0 and 1
cols = channels = ['adwords', 'facebook','linkedin','sales']

scaler = MinMaxScaler()
for col in cols:
    df[col] = scaler.fit_transform(df[[col]])

# COMMAND ----------

#adwords = df['adwords']
#facebook = df['facebook']
#linkedin = df['linkedin']
#sales = df['sales']

# COMMAND ----------

# MAGIC %md
# MAGIC ### Configure model

# COMMAND ----------

delay_channels = ['linkedin']
non_lin_channels = ['adwords','facebook']
control_vars = False
index_vars = False
outcome = 'sales'

# COMMAND ----------

# MAGIC %md
# MAGIC ### Initialize model and training: 

# COMMAND ----------

model = pm.Model()

with model:
    response_mean = []

    # channels that can have DECAY and SATURATION effects
    for channel_name in delay_channels:
        xx = df[channel_name].values

        print(f'Adding Delayed Channels: {channel_name}')
        channel_b = pm.HalfNormal(f'beta_{channel_name}',sd=5)

        alpha = pm.Beta(f'alpha_{channel_name}',alpha=3, beta=3)
        channel_mu = pm.Gamma(f'mu_{channel_name}',alpha=3,beta=1)
        response_mean.append(logistic_function(geometric_adstock_tt(xx, alpha),channel_mu) * channel_b)

    # channels that can have SATURATION effects only
    for channel_name in non_lin_channels:
        xx= df[channel_name].values

        print(f'Adding Non-linear Logistic Channel: {channel_name}')
        channel_b = pm.HalfNormal(f'beta_{channel_name}',sd=5)

        #logistic reach curve
        channel_mu = pm.Gamma(f'mu_{channel_name}', alpha=3, beta=1)
        response_mean.append(logistic_function(xx, channel_mu) * channel_b)

    # Continuous Control Variables
    if control_vars:
        for channel_name in control_vars:
            x = df[channel_name].values
            
            print(f'Adding Control: {channel_name}')
            
            control_beta = pm.Normal(f'beta_{channel_name}',sd=.25)
            channel_contrib = control_beta * x
            response_mean.append(channel_contrib)
        
    # Categorical control variables
    if index_vars:
        for var_name in index_vars:
            shape = len(df[var_name].unique())
            x = df[var_name].values
            
            print(f'Adding Index Variable: {var_name}')
            
            ind_beta = pm.Normal(f'beta_{var_name}',sd=.5,shape=shape)
            channel_contrib = ind_beta[x]
            response_mean.append(channel_contrib)

    # Noise level
    sigma = pm.Exponential('sigma',10)

    # Define likelihood
    likelihood = pm.Normal(outcome, mu=sum(response_mean), sd=sigma, observed=df[outcome].values)

# COMMAND ----------

# MAGIC %md ### TO DO - ADD MLflow 

# COMMAND ----------

# MAGIC %md 
# MAGIC ### Generate posterior predictive samples from a model given a trace:

# COMMAND ----------

with model:
    idata = pm.sample(return_inferencedata=True, target_accept=0.95)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Posterior Analysis

# COMMAND ----------

with model:
    az.plot_trace(trace);

# COMMAND ----------

az.summary(idata)

# COMMAND ----------

with model: 
    ppc = pm.sample_posterior_predictive(idata, var_names=['sales'])

# COMMAND ----------

az.plot_ppc(az.from_pymc3(posterior_predictive=ppc, model=model))
