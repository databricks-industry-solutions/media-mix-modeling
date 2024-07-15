# Databricks notebook source
# MAGIC %md This accelerator notebook is available at https://github.com/databricks-industry-solutions/media-mix-modeling. 
# MAGIC
# MAGIC To import this accelerator, please [clone the repo above into your workspace](https://docs.databricks.com/repos/git-operations-with-repos.html) instead of using the `Download .dbc` option. Please run the `RUNME` notebook at the root directory of this accelerator folder to create a cluster and a Workflow. Use the `mmm_cluster` cluster created by the RUNME notebook to run this notebook interactively.

# COMMAND ----------

# MAGIC %md 
# MAGIC
# MAGIC ## Media Mix Model
# MAGIC
# MAGIC As mentioned in the previous notebook, MMM enables companies to identify and measure the impact of their marketing campaigns across multiple channels. Now that we've simulated a dataset for daily marketing spend for three different channels and a corresponding dependent sales variable, let's see how we can use a PyMC based *media mix model* (MMM) to understand that data to help us decide what adjustments to consider, if any, to our current marketing spend. 
# MAGIC
# MAGIC But first, let's briefly discuss all the various choices we've had to make in order to even be able to write this notebook.
# MAGIC
# MAGIC <!-- insert architecture diagram -->
# MAGIC
# MAGIC *choices, choices, choices*
# MAGIC
# MAGIC 👉 **Traditional vs. Bayesian**
# MAGIC
# MAGIC There are several good open source approaches to consider for performing media mix modeling. Broadly speaking, we typically see two different approaches in use
# MAGIC
# MAGIC 1. Traditional statistical / ML models
# MAGIC 2. Bayesian modeling ✅
# MAGIC
# MAGIC Databricks works great for either approach, so you're covered whichever way you need to go. For this accelerator we'll go with a Bayesian model since we believe it drives more insights information and is more easily interpretable for decision makers and practitioners. 
# MAGIC
# MAGIC 👉 **Within Bayesian ➟ MMM Library vs. Custom Model**
# MAGIC
# MAGIC Here again there are a couple of different choices to make. 
# MAGIC
# MAGIC 1. Whether to use an MMM specific Bayesian library and if so which one
# MAGIC 2. or a general purpose Bayesian modeling framework ✅
# MAGIC
# MAGIC For this accelerator we're going to demonstrate how to create your own custom model using a general purpose Bayesian modeling framework.
# MAGIC
# MAGIC 💭 **Sidenote: MMM Libraries**
# MAGIC
# MAGIC Of the MMM specific Bayesian libraries, the ones we here about the most often are:
# MAGIC
# MAGIC 1. [PyMC-Marketing](https://github.com/pymc-labs/pymc-marketing) ❤️: An open source Bayesian marketing package supporting MMM, CLV, and other models originally from [PyMC Labs](https://www.pymc-labs.io/).
# MAGIC 2. [Robyn](https://facebookexperimental.github.io/Robyn/): An open source MMM package originally from Facebook in R.
# MAGIC 3. [Lightweight MMM](https://github.com/google/lightweight_mmm): An open source MMM package originally from Google in Python.
# MAGIC
# MAGIC If we were going to go with a library here, we'd go for PyMC-Marketing, and may do a version of this accelerator with that choice sometime in the future. We like it because it is based on PyMC, which is a Pythonic and easy to use Bayesian library that we're already using here. However, many of the techniques we'll talk about in this accelerator work for the other two equally as well, so we have you covered either way.
# MAGIC
# MAGIC 👉 **Within custom Bayesian models ➟ Which Bayesian modeling framework?**
# MAGIC
# MAGIC Instead, we're going to go with a general purpose Bayesian modeling framework. In this case, common choices we see are [RStan](https://mc-stan.org/users/interfaces/rstan), [RJags](https://cran.r-project.org/web/packages/rjags/index.html) and [MCMCPack](https://cran.r-project.org/web/packages/MCMCpack/index.html) for [R](https://www.r-project.org/) users and [PyStan](https://pystan.readthedocs.io/en/latest/), [PyJags](https://pypi.org/project/pyjags/), [PyMC](https://www.pymc.io/welcome.html) ✅, or [NumPyro](https://num.pyro.ai/en/latest/index.html#introductory-tutorials) for [Python](https://www.python.org/) users. Here, we're going to go with PyMC. It has a focus on ease of use that we admire which lets us put more focus on modeling, is [Pythonic](https://peps.python.org/pep-0020/), you don't need to learn a separate modeling language, and is performant, because it's built from the ground up on vectorized libraries. In particular, for now we're going with a somewhat older version of PyMC when it was known as PyMC3 just for compatibility reasons, but future work will upgrade this to the latest (i.e., PyMC > 5.x).
# MAGIC
# MAGIC 🎉 **Choices made: Bayesian, Custom, PyMC3 Model Chosen** ✅
# MAGIC
# MAGIC Making all these choices can be challenging! You may have different tradeoffs than us to consider, but if you're looking to accelerate your development, just going with a Bayesian approach using either PyMC like us or based on PyMC-Marketing is a great way to to speed up your MMM efforts. Since these should be already installed on the cluster if you ran the RUNME notebook, the only thing left to do is to import them and set a few other options.
# MAGIC <!--In this notebook, we showcase how to build a media mix model using the classes we wrote based on pymc3.>

# COMMAND ----------

# MAGIC %run ./config/config $reset_all_data=false

# COMMAND ----------

# MAGIC %md ### Step 1: Set up the environment
# MAGIC
# MAGIC Since we've chosen to do a custom Bayesian model using PyMC3 for our analysis, we need to import the various packages associated with that ecosystem. These packages have already been installed for you in the `config` notebook above. Note that for some packages, including this one, it may be a little difficult to find just the right configuration to get it to work properly. The configuration used here is known to work well on Databricks ML Runtime (DBR) 11.3 LTS and will be updated from time to time with newer DBR's and newer PyMC packages.

# COMMAND ----------

import pymc as pm
import arviz as az
import datetime
import numpy as np
import matplotlib.pyplot as plt
import pytensor
import pytensor.tensor as tt
import mlflow
import pickle
from pprint import pprint

import mediamix.model as mmm

print(f"Running on PyMC v{pm.__version__}")

RANDOM_SEED = 8927
np.random.seed(RANDOM_SEED)
az.style.use('arviz-darkgrid')

%config InlineBackend.figure_format = 'retina'

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 2: Load the data
# MAGIC
# MAGIC The generated dataset simulates a gold table where the input table has been transformed so the ad spend is a window leading up to the sale rather than aggregated up on the same day a sale occured. In this case, we're simply loading up the data we generated to simulate a [gold table](https://www.databricks.com/glossary/medallion-architecture), but in your system you're hopefully going to be accessing your actual gold table!
# MAGIC
# MAGIC However, you may not be there just yet. If you are at the point where you're just ingesting data from your marketing sources, then you'll want to start there, loading your data into a bronze layer, cleaning it up and creating a high quality and consistent silver layer, and then aggregating the cleansed data to produce a gold aggregate layer. The end result of that pipeline should look similar in many ways to the table we've generated here. Even though we're sort of skipping this piece by starting with a simulated gold layer, don't underestimate this piece. Getting to a good clean dataset for your analysis is an essential ingredient to success with MMM so this is a critical piece of your architecture!
# MAGIC
# MAGIC Here, we simply load our simulated gold table and have another look at it.

# COMMAND ----------

df = spark.table(gold_table_name).toPandas()
display(df)
df.plot(linewidth=0.25);

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 3: Configure the model
# MAGIC
# MAGIC Like for the generator, here we are demonstrating refactored code where the model building logic has been implemented as a `ModelConfig` class, that represents how we'd like to construct our PyMC model. Likewise, again we use the notion of a `Channel`, but in this case we're implementing the channel from the PyMC model's perspective. That is, here it's responsible for knowing how to incorporate our priors and the likelihood components for its particular channel into our overall Bayesian model.
# MAGIC
# MAGIC Feel free to copy the configuration file and adjust it to suit your needs, or even to extend or rewrite the ModelConfig class to something more tailored to your specific business case. Likewise, you might also consider exploring PyMC-Marketing as was mentioned earlier and contributing any new capabilities you develop there back to the open source community!
# MAGIC
# MAGIC Here we load up the configuration for our example and display it.

# COMMAND ----------

config_path = os.path.join(CONFIG_DIR, 'model/basic_config.yaml')
config = mmm.ModelConfig.from_config_file(config_path)
pprint(config.to_config_dict())

# COMMAND ----------

# MAGIC %md
# MAGIC #### Understanding the configuration parameters
# MAGIC
# MAGIC If you're new to Bayesian modeling, media mix modeling, or both, many of these concepts may seem overwhelming. Here are a few resources we found useful that you may want to have a look at:
# MAGIC
# MAGIC - [Bayesian Media Mix Modeling for Marketing Optimization](https://www.pymc-labs.io/blog-posts/bayesian-media-mix-modeling-for-marketing-optimization/): This is a blog post by Benjamin Vincent over at [PyMC Labs](https://www.pymc-labs.io/) based on some work they did with [HelloFresh](https://www.hellofresh.com/careers/). It does a great job of walking through some of the details of media mix modeling and understanding it from a Bayesian perspective.
# MAGIC - [A Bayesian Approach to Media Mix Modeling](https://www.youtube.com/watch?v=UznM_-_760Y): A conference presentation at [PyMCCon 2020](https://discourse.pymc.io/c/pymcon/12) from Michael Johns & Zhenyu Wang from HelloFresh that walks through many of the same concepts and how they were used there.
# MAGIC - [Bayesian Methods for Media Mix Modeling with Carryover and Shape Effects](https://research.google/pubs/pub46001/): An often cited paper that proposed this type of media mix model with flexible forms to model delay (adstock) and shape (saturation) effects.
# MAGIC
# MAGIC It can also be helpful to take advantage of ipywidgets to gain a more intuitive feel for the various shape and delay functions.

# COMMAND ----------

from mediamix import interactive as mmi
from importlib import reload
reload(mmi)

mmi.display_geometric_adstock_and_delay_interactive()

# COMMAND ----------

# MAGIC %md
# MAGIC If you experiment with the above plots, you'll find that increasing the geometric adstock \\(mu\\) parameter "spreads" the impact of the input to the function over multiple time units. From a media mix perspective, if $1 of media spend happens at \\(t=5\\) as shown, then the relative impact for that $1 can either be felt immediately \\(mu=0\\), or it can be spread over the subsequent \\(L\\) time units (the default for \\(L\\) is 12 in this case).
# MAGIC
# MAGIC For saturation, you'll find that increasing the \\(mu\\) parameter yields a saturation effect where an additional $1 of media spend does not yield a linear increase of the impact as the impact itself grows larger. For a very low saturation value, it may be close to linear, but as \\(mu\\) increases then $1 gets you less and less of an effect as the overall amount spent continues to grow.
# MAGIC
# MAGIC In a Bayesian model, we specify that we believe whether each channel would exhibit these effects, and if so, the valid range of our prior beliefs, via our priors and likelihood functions. Diving deeper into the details of Bayesian modeling is beyond the scope of this accelerator and is well served by the resources we mentioned earlier, along with many other books. But hopefully this brief explanation gives you a little more insight into what's going on under the hood here.

# COMMAND ----------

# MAGIC %md ### Step 4: Run inference
# MAGIC
# MAGIC Now that we've defined our model structure, we can run inference to analyze the posterior distributions of the parameters of interest! Here, we can also specify various settings of the inference process itself, such as the number of draws to collect (per core / chain) and the number of warmup (i.e., tune) draws to take before we start collecting them.
# MAGIC
# MAGIC As you experiment with these parameters and others, how do you keep track of which parameters and model structure settings gave you the best result? The answer: [MLflow](https://mlflow.org/) of course! MLflow makes it easy to keep track of all of your experiment runs, including any artifacts, metrics, parameters and other metadata you might want to capture. Here, we're keeping track of `params` below, which is super easy to log as a dictionary:
# MAGIC
# MAGIC     mlflow.log_params(params)
# MAGIC
# MAGIC A separate experiment object called *Media Mix Modeling* has been created in your home folder already as part of the `config` notebook in this solution accelerator, and its already been linked to this notebook, so if you expand the experiment tab you'll see a new entry every time you run the `run_inference` command below.
# MAGIC
# MAGIC Also, note that we are also logging the configuration file we used to specify the model choices and configuration parameters. This is one of the main reasons we built some helper functions to load and save the model structure to a yaml file in this manner; it makes it really easy to log it as an artifact and then reproduce it later as needed!

# COMMAND ----------

params = {
    'draws': 1000,
    'tune': 1000,
    'init': 'auto'}

model, idata, scalers = config.run_inference(params, df)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 5: Analyze the results
# MAGIC
# MAGIC Now that we've run inference and have our `idata` or inference data object back from PyMC, we can analyze it to help with our media mix decision process!
# MAGIC
# MAGIC There are two or three parameters of interest here across the three channels we're analyzing in this scenarios:
# MAGIC
# MAGIC | Channel         | Beta             | Saturation    | Adstock (geometric) |
# MAGIC | --------------- | ---------------- | ------------- | ------------------- |
# MAGIC | Facebook        | X                | X             |                     |
# MAGIC | Adwords         | X                | X             |                     |
# MAGIC | LinkedIn        | X                | X             | X                   |
# MAGIC
# MAGIC We also model an intercept parameter and sigma to capture the bias and noise associated with the model vs our target sales variable.
# MAGIC
# MAGIC The saturation parameter tells us how saturated a given channel is. This tells us the relative efficiency of investing more into a particular channel, and we'd interpret that as follows:
# MAGIC
# MAGIC   * Low saturation = room to grow
# MAGIC   * High saturation = could taper back a bit
# MAGIC
# MAGIC We can use saturation to find the optimal spend for a given channel. In the example scenario we've modeled below, we see that LinkedIn is has relatively less saturation than the other two channels, and can incorporate that into our investing decision.
# MAGIC
# MAGIC For LinkedIn we are also assuming that there may be a residual effect to our spend in this channel, which we've modeled by applying a geometric adstock function in our model. A long decay means we could spend a little less during a given week and extract extra value and efficiency (i.e., more bang for the buck)!
# MAGIC
# MAGIC Lastly we look at the \\(\beta\\) parameter for each channel, which models the relative impact on the dependent variable, in this case sales. Here, AdWords has \\(\beta \approx 1.5 \\), Facebook has \\(\beta \approx 1.0 \\), and LinkedIn has \\(\beta \approx 2.5 \\). 
# MAGIC
# MAGIC In this (simulated) scenario, our analysis would tell us that we may want to allocate more of our budget to LinkedIn, since it is showing as both less saturated *and* having a higher relative impact on sales than the other two channels in our mix!

# COMMAND ----------

az.summary(idata)

# COMMAND ----------

# MAGIC %md
# MAGIC We can also compare how seeing this particular dataset has caused us to update our prior beliefs. Let's take our simulated LinkedIn as an example. Here, we used the rather uninformative priors of a Beta distribution with parameters \\(\alpha=1\\) and \\(\beta=3\\) for our geometric adstock \\(\alpha\\) parameter and a Gamma distribution with parameters \\(\alpha=3\\) and \\(\beta=1\\) for our saturation \\(\mu\\) parameter. We also have priors for the \\(\beta\\) parameter and others, but let's focus on these two for the sake of simplicity. If we plot these two prior distributions and compare them to the histogram for our traces for these two parameters from our inference run, we can see how much our beliefs have shifted based on the data we've seen.
# MAGIC
# MAGIC As you can see in the plots, because we simulated a relatively large amount of synthetic data for this example, our beliefs have been updated rather dramatically!

# COMMAND ----------

with model:
    idata.extend(pm.sample_prior_predictive())
    
az.plot_dist_comparison(idata, var_names=["saturation_linkedin", "geometric_adstock_linkedin"], figsize=(12, 8));

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 6: Inspect the trace visually
# MAGIC
# MAGIC You may have noticed in the table above that each parameter has multiple statistics associated with it, rather than just a single point estimate. This is because Bayesian inference gives us an *entire posterior distribution* to consider rather than a single point estimate! We can see this visually by looking at the trace plot as showing below! For each parameter, we can see not only where the posterior beliefs of our model are centered, but also the spread. In other words, saying that our posterior beliefs having seen our data for \\(\beta\\) for adwords is \\(1.5 \pm 0.1 \\) is a lot different than saying \\(1.5 \pm 1.0 \\). Gaining a clear and intuitive understanding of the uncertainty in our posterior beliefs is one of the key advantages of using a Bayesian approach, and it's important to consider it when determining your advertising spend.
# MAGIC
# MAGIC The traceplot gives us both a density plot for each parameter based on the traces from our inference, as well as the trace values themselves plotted over the draws. In general, we want those traces on the right to look like "fuzzy caterpillars". If they don't, you will need to revisit your model configuration. This, along with the reported *effective sample size (ESS)* and *r_hat* can be used to ensure your model has converged. Have a look at a good Bayesian modeling text or one of the many good resources or documentation sets online for more information.

# COMMAND ----------

az.plot_trace(idata);

# COMMAND ----------

# MAGIC %md 
# MAGIC ### Step 7: Inspect posterior predictive samples
# MAGIC
# MAGIC There are many other plots that may be useful to produce as well. As another example, below we've produced our posterior predictive sales plot. Ideally, posterior predictive samples would align closely to our observed sales data, which in this case they do!
# MAGIC
# MAGIC In practice, you will want plots that:
# MAGIC
# MAGIC - help you assess whether your model has converged and correctly describes your data
# MAGIC - understand the relative distributions of the parameters themselves and their interpretation
# MAGIC - tie things back to your specific business context and aid in decision making
# MAGIC
# MAGIC The point of showing these plots here in the context of Databricks and MLflow is to prompt ideas for how you can capture these figures as artifacts as part of your experimentation process so you can easily keep track of them over time and across your team. Being able to produce these diagrams in a one-off fashion can be helpful, but standardizing on a set of artifacts across your team and collecting them systematically as we've shown here can really take your practice to the next level!

# COMMAND ----------

with model:
    idata.extend(pm.sample_posterior_predictive(idata.posterior))
az.plot_ppc(idata);

# COMMAND ----------

# MAGIC %md
# MAGIC ### Next steps
# MAGIC
# MAGIC Congratulations! You now know how to build a custom Bayesian model to analyze your adspend data and interpret the resulting inference data! Now that you have a good understanding of the models associated with your past performance of your adspend, you're in a much better position to make better investment decisions regarding your advertising budget allocation.
# MAGIC
# MAGIC Another point worth noting is that we were able to recover the parameters we specified in our simulation using our model. This shouldn't be too surprising, but the power of this approach should not be underestimated. When building a Bayesian model, it can be quite easy to specify the model incorrectly, either the prior distributions used, the associated prior parameters, the structure of the likelihood function, etc... in such a way that you aren't actually able to recover your parameters. To make matters worse, you won't know for sure whether the problem is some bug in your model specification or some nuanced things about your data you're missing, or even data quality issues. Having a generator or simulation that you can control can really help ensure your model can at least work vs a simulated dataset, while other aspects of the Databricks platform can help with managing data quality and understanding the data itself.
# MAGIC
# MAGIC There are a couple of different ways to progress from here:
# MAGIC
# MAGIC 1. You can feed this information into downstream optimization algorithms to explore various investment allocations and their expected return based on the output of this analysis.
# MAGIC 2. You can break down high level analysis by different slices like brand or geographical region. This is where having your analysis code report back to MLflow and using a distributed compute platform like Databricks can really come in handy!
# MAGIC 3. You can productionize this analysis using multitask jobs and write out the inference results to a new set of gold tables from which you can build dashboards for other marketing personnel.
# MAGIC 4. You can further wrap this analysis using a notebook as a UI for analysts to be able to easily experiment with various modeling configurations easily.
# MAGIC
# MAGIC Hopefully this solution accelerator will help you build a strong foundation for your media mix modeling practice!
