![image](https://github.com/databricks-industry-solutions/.github/raw/main/profile/solacc_logo_wide.png)

[![CLOUD](https://img.shields.io/badge/CLOUD-ALL-blue?logo=googlecloud&style=for-the-badge)](https://cloud.google.com/databricks)
[![POC](https://img.shields.io/badge/POC-10_days-green?style=for-the-badge)](https://databricks.com/try-databricks)


## Media Mix Modeling Accelerator

**Under construction: We're just finishing up some changes to convert his solution accelerator to use PyMC-Marketing. Thanks for your interest and please be sure to check back again in a few days for when the changes are completed!**

MMM (Marketing or Media Mix Modeling), is a data-driven methodology that enables companies to identify and measure the impact of their marketing campaigns across multiple channels.  MMM helps businesses make better-informed decisions about their advertising and marketing strategies. 

By analyzing data from various channels such as TV, social media, email marketing, and more, MMM can determine which channels are contributing the most to strategic KPIs, such as sales. By including external events and indicators, decision makers can better understand the impact of outside factors (such as holidays, economic conditions, or weather) and avoid accidently over-valuing the impact of ad spend alone.

Databricks Lakehouse offers a unified platform for building modernized MMM solutions that are scalable and flexible. Marketers can unifies various upstream data sources, automates data ingestion, processing, and transformation, and provides full transparency and traceability of data usage. With powerful DSML capabilities and collaborative workstream, marketers can seamlessly leverage the full potential of their data, driving more informed and effective marketing investment decisions. 


___
<corey.abshire@databricks.com>,  <tristan.nixon@databricks.com>,  <layla@databricks.com>

___

<arch image>

___

&copy; 2022 Databricks, Inc. All rights reserved. The source in this notebook is provided subject to the Databricks License [https://databricks.com/db-license-source].  All included or referenced third party libraries are subject to the licenses set forth below.

| library                                | description             | license    | source                                              |
|----------------------------------------|-------------------------|------------|-----------------------------------------------------|
| pytest                                 | Python testing framework      | MIT        | https://github.com/pytest-dev/pytest                      |
| pymc3 | Probabilistic Programming in Python | Apache 2.0 | https://github.com/pymc-devs/pymc |

## Getting started

Although specific solutions can be downloaded as .dbc archives from our websites, we recommend cloning these repositories onto your databricks environment. Not only will you get access to latest code, but you will be part of a community of experts driving industry best practices and re-usable solutions, influencing our respective industries. 

<img width="500" alt="add_repo" src="https://user-images.githubusercontent.com/4445837/177207338-65135b10-8ccc-4d17-be21-09416c861a76.png">

To start using a solution accelerator in Databricks simply follow these steps: 

1. Clone solution accelerator repository in Databricks using [Databricks Repos](https://www.databricks.com/product/repos)
2. Attach the `RUNME` notebook to any cluster and execute the notebook via Run-All. A multi-step-job describing the accelerator pipeline will be created, and the link will be provided. The job configuration is written in the RUNME notebook in json format. 
3. Execute the multi-step-job to see how the pipeline runs. 
4. You might want to modify the samples in the solution accelerator to your need, collaborate with other users and run the code samples against your own data. To do so start by changing the Git remote of your repository  to your organization’s repository vs using our samples repository (learn more). You can now commit and push code, collaborate with other user’s via Git and follow your organization’s processes for code development.

The cost associated with running the accelerator is the user's responsibility.


## Project support 

Please note the code in this project is provided for your exploration only, and are not formally supported by Databricks with Service Level Agreements (SLAs). They are provided AS-IS and we do not make any guarantees of any kind. Please do not submit a support ticket relating to any issues arising from the use of these projects. The source in this project is provided subject to the Databricks [License](./LICENSE). All included or referenced third party libraries are subject to the licenses set forth below.

Any issues discovered through the use of this project should be filed as GitHub Issues on the Repo. They will be reviewed as time permits, but there are no formal SLAs for support. 
