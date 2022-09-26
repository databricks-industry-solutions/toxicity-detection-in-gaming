# Databricks notebook source
# MAGIC %pip install kaggle spark-nlp==4.0.0

# COMMAND ----------

import re
import os
import json

# COMMAND ----------

username = dbutils.notebook.entry_point.getDbutils().notebook().getContext().userName().get().split('@')[0]
username_sql = re.sub('\W', '_', username)
tmpdir = f"/dbfs/tmp/{username}/"
tmpdir_dbfs = f"/tmp/{username}"
database_name = f"gaming_{username_sql}"
database_location = f"{tmpdir}gaming"

# COMMAND ----------

import os
# os.environ['kaggle_username'] = 'YOUR KAGGLE USERNAME HERE' # replace with your own credential here temporarily or set up a secret scope with your credential
os.environ['kaggle_username'] = dbutils.secrets.get("solution-accelerator-cicd", "kaggle_username")

# os.environ['kaggle_key'] = 'YOUR KAGGLE KEY HERE' # replace with your own credential here temporarily or set up a secret scope with your credential
os.environ['kaggle_key'] = dbutils.secrets.get("solution-accelerator-cicd", "kaggle_key")

os.environ['tmpdir'] = tmpdir

# COMMAND ----------

spark.sql(f"CREATE DATABASE IF NOT EXISTS {database_name} location '{database_location}'")
spark.sql(f"USE {database_name}")

# COMMAND ----------


