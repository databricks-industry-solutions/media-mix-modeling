# Databricks notebook source
# MAGIC %pip install --no-deps typing-extensions==3.7.4 deprecat==2.1.1 semver==2.13.0 cftime==1.6.2 \
# MAGIC     fastprogress==0.2.0 netCDF4==1.6.2 xarray-einstats==0.3 xarray==0.21.0 theano-pymc==1.1.2 \
# MAGIC     arviz==0.11.0 pymc3==3.11.5

# COMMAND ----------

import pytest
import os
import sys

repo_name = 'media-mix-modeling'

# Get the path to this notebook, for example "/Workspace/Repos/{username}/{repo-name}".
notebook_path = dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get()

# Get the repo's root directory name.
repo_root = os.path.dirname(os.path.dirname(notebook_path))

# Prepare to run pytest from the repo.
os.chdir(f"/Workspace/{repo_root}/{repo_name}")
print(os.getcwd())

# Skip writing pyc files on a readonly filesystem.
sys.dont_write_bytecode = True

# Run pytest.
retcode = pytest.main([".", "-v", "-p", "no:cacheprovider"])

# Fail the cell execution if there are any test failures.
assert retcode == 0, "The pytest invocation failed. See the log for details."
