# Databricks notebook source
# MAGIC %md This accelerator notebook is available at https://github.com/databricks-industry-solutions/media-mix-modeling. Please use the `mmm_cluster` cluster created by RUNME notebook at the root directory of this accelerator folder, if you run this notebook interactively.

# COMMAND ----------

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

# MAGIC %md
# MAGIC ### Configure environment

# COMMAND ----------

import pymc3 as pm
import arviz as az
import datetime
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import theano
import theano.tensor as tt

from mediamix.transforms import saturation, geometric_adstock

print(f"Running on PyMC3 v{pm.__version__}")

RANDOM_SEED = 8927
np.random.seed(RANDOM_SEED)
az.style.use('arviz-darkgrid')

%config InlineBackend.figure_format = 'retina'

# COMMAND ----------

# MAGIC %md
# MAGIC ### Load the data
# MAGIC 
# MAGIC The generated dataset simulates a gold table where the input table has been
# MAGIC transformed so the ad spend is a window leading up to the sale rather than 
# MAGIC aggregated up on the same day a sale occured.

# COMMAND ----------

df = spark.table(gold_table_name).toPandas()
display(df)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Configure model

# COMMAND ----------

# define the features to be used in the model
delay_channels = ['linkedin']
non_lin_channels = ['adwords','facebook']
channels = non_lin_channels + delay_channels
control_vars = None
index_vars = None
outcome = 'sales'

# COMMAND ----------

# MAGIC %md
# MAGIC ### Scale the data

# COMMAND ----------

# scale variables to be between 0 and 1
scalers = {}
for c in channels:
    scalers[c] = MinMaxScaler()
    df[c] = scalers[c].fit_transform(df[[c]])

# custom scaling on outcome
custom_outcome_scale = 100000
df[outcome] /= custom_outcome_scale

# COMMAND ----------

# visualize the scaled results
for c in channels + [outcome]:
    plt.plot(df[c], label=f'{c}', linewidth=0.25)
plt.legend();

# COMMAND ----------

# MAGIC %md
# MAGIC ### Initialize model and training

# COMMAND ----------

model = pm.Model()

with model:
    response_mean = []

    intercept = pm.Normal('intercept', mu=0, sd=10)
    response_mean.append(intercept)

    # channels that can have DECAY and SATURATION effects
    for channel_name in delay_channels:
        xx = df[channel_name].values

        print(f'Adding Delayed Channels: {channel_name}')
        channel_b = pm.HalfNormal(f'beta_{channel_name}', sd=1)

        alpha = pm.Beta(f'alpha_{channel_name}', alpha=1, beta=3)
        channel_mu = pm.Gamma(f'mu_{channel_name}', alpha=3, beta=1)
        response_mean.append(saturation(geometric_adstock(xx, alpha),channel_mu) * channel_b)

    # channels that can have SATURATION effects only
    for channel_name in non_lin_channels:
        xx= df[channel_name].values

        print(f'Adding Non-linear Logistic Channel: {channel_name}')
        channel_b = pm.HalfNormal(f'beta_{channel_name}', sd=1)

        #logistic reach curve
        channel_mu = pm.Gamma(f'mu_{channel_name}', alpha=3, beta=1)
        response_mean.append(saturation(xx, channel_mu) * channel_b)

    # Continuous Control Variables
    if control_vars:
        for channel_name in control_vars:
            x = df[channel_name].values
            
            print(f'Adding Control: {channel_name}')
            
            control_beta = pm.Normal(f'beta_{channel_name}', sd=1)
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
    sigma = pm.HalfCauchy('sigma', 5)

    # Define likelihood
    likelihood = pm.Normal(outcome, mu=sum(response_mean), sd=sigma, observed=df[outcome].values)

# COMMAND ----------

# MAGIC %md ### TO DO - ADD MLflow 

# COMMAND ----------

with model:
    idata = pm.sample(return_inferencedata=True)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Posterior Analysis

# COMMAND ----------

az.summary(idata)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Inspect the trace visually

# COMMAND ----------

with model:
    az.plot_trace(idata);

# COMMAND ----------

# MAGIC %md 
# MAGIC ### Inspect posterior predictive samples

# COMMAND ----------

with model: 
    ppc = pm.sample_posterior_predictive(idata, var_names=['sales'])

# COMMAND ----------

az.plot_ppc(az.from_pymc3(posterior_predictive=ppc, model=model));

# COMMAND ----------

# MAGIC %md 
# MAGIC ### Reference
# MAGIC [From: Introduction to PyMC3](https://github.com/junpenglao/PrecisionWorkshop1_Prep/blob/master/notebooks/1a.%20Introduction%20to%20PyMC3.ipynb)
# MAGIC * Gelman et al. [3] break down the business of Bayesian analysis into three primary steps:
# MAGIC * Specify a full probability model, including all parameters, data, transformations, missing values and predictions that are of interest.
# MAGIC * Calculate the posterior distribution of the unknown quantities in the model, conditional on the data.
# MAGIC * Perform model checking to evaluate the quality and suitablility of the model.
