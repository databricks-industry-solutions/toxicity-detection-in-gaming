# Databricks notebook source
# MAGIC %md 
# MAGIC You may find this series of notebooks at https://github.com/databricks-industry-solutions/toxicity-detection-in-gaming and more information about this solution accelerator at https://www.databricks.com/solutions/accelerators/toxicity-detection-for-gaming.

# COMMAND ----------

# MAGIC %md
# MAGIC <div >
# MAGIC   <img src="https://cme-solution-accelerators-images.s3-us-west-2.amazonaws.com/toxicity/solution-accelerator-logo.png"; width="50%">
# MAGIC </div>

# COMMAND ----------

# MAGIC %md
# MAGIC ## Overview
# MAGIC
# MAGIC Toxicity can have a large impact on player engagement and satisfaction. Game companies are working on ways to address forms of toxicity in their platforms. One of the most common interactions with toxicity is in chat boxes or in-game messaging systems. As companies are becoming more data driven, the opportunity to detect toxicity using the data at hand is present, but technically challenging. This solution accelerator is a head start on deploying a ML-enhanced data pipeline to address toxic messages in real time.
# MAGIC
# MAGIC ** Authors**
# MAGIC - Duncan Davis [<duncan.davis@databricks.com>]
# MAGIC - Dan Morris [<dan.morris@databricks.com>]

# COMMAND ----------

# MAGIC %md
# MAGIC ## About This Series of Notebooks
# MAGIC
# MAGIC * This series of notebooks is intended to help you use multi-label classification to detect and analyze toxicity in your data.
# MAGIC
# MAGIC * In support of this goal, we will:
# MAGIC  * Load toxic-comment training data from Jigsaw and game data from Dota 2.
# MAGIC  * Create one pipeline for streaming and batch to detect toxicity in near real-time and/or on an ad-hoc basis. This pipeline can then be used for managing tables for reporting, ad hoc queries, and/or decision support.
# MAGIC  * Label text chat data using Multi-Label Classification.
# MAGIC  * Create a dashboard for monitoring the impact of toxicity.

# COMMAND ----------

# MAGIC %md
# MAGIC ## About the Data

# COMMAND ----------

# MAGIC %md
# MAGIC #### Jigsaw Dataset
# MAGIC
# MAGIC * The dataset used in this accelerator is from [Jigsaw](https://jigsaw.google.com/). Jigsaw is a unit within Google that does work to create a safer internet. Some of the areas that Jigsaw focuses on include: disinformation, censorship, and toxicity.
# MAGIC
# MAGIC * Jigsaw posted this dataset on [Kaggle](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/data) three years ago for the toxic comment classification challenge. This is a multilabel classification problem that includes the following labels:
# MAGIC   * Toxic, Severe Toxic, Obscene, Threat, Insult, and Identity Hate
# MAGIC   
# MAGIC * Further details about this dataset
# MAGIC   * Dataset title: Jigsaw Toxic Comment Classification Challenge
# MAGIC   * Dataset source URL: https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/data
# MAGIC   * Dataset source description: Kaggle competition
# MAGIC   * Dataset license: please see dataset source URL above

# COMMAND ----------

# MAGIC %md
# MAGIC #### DOTA 2 Matches Dataset
# MAGIC [Dota 2](https://blog.dota2.com/?l=english)
# MAGIC
# MAGIC <div >
# MAGIC   <img src="https://cme-solution-accelerators-images.s3-us-west-2.amazonaws.com/toxicity/dota_2.jpg"; width="20%">
# MAGIC </div>
# MAGIC
# MAGIC This dataset is from is a multiplayer online battle arena (MOBA) video game developed and published by Valve. 
# MAGIC
# MAGIC Dota 2 is played in matches between two teams of five players, with each team occupying and defending their own separate base on the map.
# MAGIC
# MAGIC Further details about this dataset
# MAGIC   * Dataset title: Dota 2 Matches
# MAGIC   * Dataset source URL: https://www.kaggle.com/devinanzelmo/dota-2-matches
# MAGIC   * Dataset source description: Kaggle competition
# MAGIC   * Dataset license: please see dataset source URL above

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 1: Configure the Environment
# MAGIC
# MAGIC In this step, we will:
# MAGIC   1. Install the Kaggle library
# MAGIC   2. Obtain KAGGLE_USERNAME and KAGGLE_KEY for authentication

# COMMAND ----------

# MAGIC %md
# MAGIC ##### 1.1: Install the Kaggle library

# COMMAND ----------

# MAGIC %run ./config/notebook_config

# COMMAND ----------

# MAGIC %md
# MAGIC ##### 1.2: Obtain KAGGLE_USERNAME and KAGGLE_KEY for authentication
# MAGIC
# MAGIC * Instructions on how to obtain this information can be found [here](https://www.kaggle.com/docs/api).

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 2: Download the data
# MAGIC
# MAGIC In this step, we will:
# MAGIC   1. Download the Jigsaw dataset
# MAGIC   2. Unzip the Jigsaw dataset
# MAGIC   3. Download the Dota 2 game dataset
# MAGIC   4. Unzip the Dota 2 game dataset

# COMMAND ----------

# MAGIC %md
# MAGIC ##### 2.1: Download jigsaw dataset
# MAGIC The Conversation AI team, a research initiative founded by Jigsaw and Google (both a part of Alphabet) are working on tools to help improve online conversation. One area of focus is the study of negative online behaviors, like toxic comments (i.e. comments that are rude, disrespectful, or otherwise likely to make someone leave a discussion).

# COMMAND ----------

# MAGIC %sh -e
# MAGIC mkdir -p /root/.kaggle/
# MAGIC
# MAGIC echo """{\"username\":\"$kaggle_username\",\"key\":\"$kaggle_key\"}""" > /root/.kaggle/kaggle.json
# MAGIC
# MAGIC chmod 600 /root/.kaggle/kaggle.json

# COMMAND ----------

# MAGIC %sh -e
# MAGIC rm -rf $tmpdir
# MAGIC mkdir -p $tmpdir
# MAGIC
# MAGIC kaggle competitions download -p "$tmpdir" -c jigsaw-toxic-comment-classification-challenge 

# COMMAND ----------

# MAGIC %md
# MAGIC ##### 2.2: Unzip jigsaw dataset
# MAGIC Breakdown of the data downloaded: https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/overview

# COMMAND ----------

# MAGIC %sh -e
# MAGIC cd $tmpdir
# MAGIC unzip -o jigsaw-toxic-comment-classification-challenge.zip
# MAGIC unzip -o train.csv.zip
# MAGIC unzip -o test.csv.zip

# COMMAND ----------

# MAGIC %md
# MAGIC ##### 2.3: Download Dota 2 dataset
# MAGIC Dota 2 is a multiplayer online battle arena (MOBA) video game in which two teams of five players compete to collectively destroy a large structure defended by the opposing team known as the "Ancient", whilst defending their own.

# COMMAND ----------

# MAGIC %sh -e
# MAGIC kaggle datasets download -p $tmpdir -d devinanzelmo/dota-2-matches

# COMMAND ----------

# MAGIC %md
# MAGIC ##### 2.4: Unzip Dota 2 dataset
# MAGIC Breakdown of the data included in the download: https://www.kaggle.com/devinanzelmo/dota-2-matches

# COMMAND ----------

# MAGIC %sh -e
# MAGIC cd $tmpdir
# MAGIC unzip -o dota-2-matches.zip 

# COMMAND ----------

# MAGIC %md
# MAGIC ## Next Steps
# MAGIC * In the next notebook, we will load the data we generated here into [Delta](https://docs.databricks.com/delta/delta-intro.html) tables.

# COMMAND ----------


