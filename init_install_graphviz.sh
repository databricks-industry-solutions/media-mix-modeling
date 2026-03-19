#!/bin/bash
# Init script to install the Graphviz system library on Databricks clusters.
# Required for DAG visualization in the fit_model notebook.
sudo apt-get install -y graphviz
