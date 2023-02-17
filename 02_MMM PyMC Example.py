# Databricks notebook source
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

!pip install pymc3==3.11.5 

# COMMAND ----------

import pymc3 as pm
import arviz as az
import datetime
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import theano
import theano.tensor as tt
# from databricks import koalas as ks

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
# MAGIC need to adjust the input table so the ad spend is a window leading up to the sale rather than aggregated up on the same day a sale occured

# COMMAND ----------

df = spark.sql('''
select
  date
  , 'adwords' as platform
  , sum(cost) as cost
from marketing_adwords.ad_stats
group by 1

union all

select
  date(day) as date
  , 'linkedin' as platform
  , sum(cost_in_usd) as cost
from marketing_linkedin_ads.ad_analytics_by_campaign
group by 1

union all

select
  date(close_date) as date
  , 'sales' as platform
  , sum(amount) as cost
from marketing_sfdc_fivetran.opportunity
where is_closed = True
  and is_won = True
group by 1

union all

select
  date
  , 'facebook' as platform
  , sum(spend) as cost
from marketing_facebook_ads.basic_campaign
group by 1
''')

# COMMAND ----------

df = df.to_koalas().pivot_table(
  index=['date'],
  columns='platform',
  values='cost',
  aggfunc='sum',
).fillna(0).reset_index()

# COMMAND ----------

display(df)

# COMMAND ----------

(df
  .to_spark()
  .write
  .format("delta")
  .mode('overwrite')
  .save("/home/layla.yang@databricks.com/fivetran")
)


# COMMAND ----------

# MAGIC %sql
# MAGIC 
# MAGIC CREATE or replace TABLE layla.fivetran
# MAGIC USING delta
# MAGIC AS SELECT *
# MAGIC FROM delta.`/home/layla.yang@databricks.com/fivetran`
# MAGIC ;
# MAGIC 
# MAGIC select * from layla.fivetran limit 10

# COMMAND ----------

df = spark.table('layla.fivetran').toPandas()

# COMMAND ----------

import datetime
# Filtered for initial analysis
df = df[df['date'] >= datetime.date(2020,1,1)].sort_values('date')

# COMMAND ----------

from sklearn.preprocessing import MinMaxScaler
# scale variables to be between 0 and 1
cols = channels = ['adwords', 'facebook','linkedin','sales']

scaler = MinMaxScaler()
for col in cols:
  df[col] = scaler.fit_transform(df[[col]])

# COMMAND ----------

adwords = df['adwords']
facebook = df['facebook']
linkedin = df['linkedin']
sales = df['sales']

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

# Geometric Adstock Function
def geometric_adstock_tt(x,alpha=0,L=12,normalize=True):
  '''
  :param alpha: rate of decay (float)
  :param L: Length of time carryover effects can have an impact (int)
  :normalize: Boolean
  :return transformed spend vector
  '''
  
  w = tt.as_tensor_variable([tt.power(alpha,i) for i in range(L)])
  xx = tt.stack([tt.concatenate([tt.zeros(i), x[:x.shape[0] - i]]) for i in range(L)])
  
  if not normalize:
    y = tt.dot(w,xx)
  else:
    y = tt.dot(w / tt.sum(w), xx)
  return y

# Nonlinear Saturation Function
def logistic_function(x_t, mu=0.1):
  """
  :param x_t: marketing spend vector (float)
  :param mu: half-saturation point (float)
  :return transformed spend vector
  """
  
  return (1 - np.exp(-mu * x_t)) / (1 + np.exp(-mu * x_t))

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
  trace = pm.sample(return_inferencedata=False,target_accept = 0.95)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Posterior Analysis

# COMMAND ----------

with model:
    az.plot_trace(trace);

# COMMAND ----------

az.summary(trace)

# COMMAND ----------

az.summary(trace)

# COMMAND ----------

from pymc3 import summary
summary(trace['beta_linkedin'])

# COMMAND ----------

summary(trace)
