# Databricks notebook source
# MAGIC %md 
# MAGIC You may find this series of notebooks at https://github.com/databricks-industry-solutions/toxicity-detection-in-gaming and more information about this solution accelerator at https://www.databricks.com/solutions/accelerators/toxicity-detection-for-gaming.

# COMMAND ----------

# MAGIC %md
# MAGIC <div style="text-align: left">
# MAGIC   <img src="https://cme-solution-accelerators-images.s3-us-west-2.amazonaws.com/toxicity/solution-accelerator-logo.png"; width="50%">
# MAGIC </div>

# COMMAND ----------

# MAGIC %md
# MAGIC ## Overview
# MAGIC 
# MAGIC ### In this lesson you:
# MAGIC * Load a model from MLflow
# MAGIC * Productionalize a streaming & batch inference pipeline
# MAGIC * Explore the impact of toxicity

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 1: Configure the Environment
# MAGIC 
# MAGIC Import libraries needed

# COMMAND ----------

# MAGIC %run ./config/notebook_config

# COMMAND ----------

from pyspark.sql.functions import col, struct
from pyspark.sql.types import *
import mlflow

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 2: Load classification model from MLflow Model Registry
# MAGIC 
# MAGIC The MLflow Model Registry component is a centralized model store, a set of APIs, and UI, used to collaboratively manage the full lifecycle of an MLflow Model. It provides model lineage (e.g. which MLflow experiment and run produced the model), model versioning, stage transitions (e.g. from staging to production), and annotations.
# MAGIC 
# MAGIC [Documentation](https://www.mlflow.org/docs/latest/model-registry.html#using-the-model-registry)

# COMMAND ----------

model_name='Toxicity MultiLabel Classification'
stage = None

model = mlflow.spark.load_model(f'models:/{model_name}/{stage}')

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 3: Productionalizing ML Pipelines with Batch or Streaming
# MAGIC 
# MAGIC Note: in this solution accelerator, we are storing the data back into Delta Lake, but we could just as easily push out events or alerts based on the inference results.
# MAGIC <div>
# MAGIC   <img src="https://cme-solution-accelerators-images.s3-us-west-2.amazonaws.com/toxicity/data-pipelines.png"; width="50%" />
# MAGIC /<div>
# MAGIC ##### Structured Streaming for One API that handles Batch & Streaming
# MAGIC 
# MAGIC [Structured Streaming](https://spark.apache.org/docs/latest/structured-streaming-programming-guide.html) is a scalable and fault-tolerant stream processing engine built on the Spark SQL engine. You can express your streaming computation the same way you would express a batch computation on static data. The Spark SQL engine will take care of running it incrementally and continuously and updating the final result as streaming data continues to arrive. You can use the Dataset/DataFrame API in Scala, Java, Python or R to express streaming aggregations, event-time windows, stream-to-batch joins, etc.
# MAGIC 
# MAGIC [Here](https://docs.databricks.com/spark/latest/structured-streaming/index.html) is a link that shows examples and code for streaming with Kafka, Kinesis, and other popular sources.

# COMMAND ----------

# MAGIC %md
# MAGIC ##### 3.1 Read Stream

# COMMAND ----------

raw_comments = spark.readStream.format("Delta")\
  .table("Toxicity_Chat")\
  .withColumnRenamed('key', 'comment_text')\
  .repartition(5000)

# COMMAND ----------

# MAGIC %md
# MAGIC ##### 3.2 Inference on streaming dataframe

# COMMAND ----------

comments_pred = model.transform(raw_comments)\
  .withColumnRenamed('key', 'comment_text')\
  .drop('document', 'token', 'universal_embeddings')\
  .withColumn('predicted',col('class.result'))

# COMMAND ----------

# MAGIC %md
# MAGIC ##### 3.3 Write Stream
# MAGIC For the sake of the accelerator, we clean up any previous checkpoints and start the stream. We write the output of comments_pred to the delta table "Toxicity_Chat_Pred"

# COMMAND ----------

# Initialize a user-specific checkpoint
user = dbutils.notebook.entry_point.getDbutils().notebook().getContext().tags().apply('user')
checkpoint = f"/toxicity_accelerator/{user}/_checkpoints/Toxicity_Chat_Pred"
dbutils.fs.rm(checkpoint, True)

# COMMAND ----------

# The trigger makes the Structured Streaming pipeline run once
comments_pred.writeStream\
  .trigger(once=True)\
  .format("Delta")\
  .option("checkpointLocation", checkpoint)\
  .option("mergeSchema", "true")\
  .table("Toxicity_Chat_Pred") \
  .awaitTermination() # set awaitTermination to block subsequent blocks from execution

# COMMAND ----------

# MAGIC %md
# MAGIC ##### 3.4 Dataframe API
# MAGIC The dataframe api is an optional way to do batch inference. The below cell will recreate the same results as the streaming job above.

# COMMAND ----------

# MAGIC %sql
# MAGIC DROP TABLE IF EXISTS Toxicity_Chat_Pred

# COMMAND ----------

chatDF = spark.table("Toxicity_Chat").withColumnRenamed('key', 'comment_text').repartition(5000)
chatDF = model.transform(chatDF)\
  .withColumn('predicted',col('class.result'))\
  .drop('document', 'token', 'universal_embeddings', 'class')

chatDF.write.format("delta").mode("overwrite").saveAsTable("Toxicity_Chat_Pred")

# COMMAND ----------

# MAGIC %md
# MAGIC ##### 3.5 Display new table with inferred labels

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT match_id, comment_text, slot, time, unit, predicted FROM Toxicity_Chat_Pred LIMIT 1

# COMMAND ----------

# MAGIC %md
# MAGIC Now that we have the pipeline and the silver table with the predicted labels, we can move onto combing the labeled data with our game data
# MAGIC 
# MAGIC <div><img src="https://cme-solution-accelerators-images.s3-us-west-2.amazonaws.com/toxicity/delta-lake-silver.png"; width="50%" /></div>

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 4: Exploring the impact of toxicity on 50K Dota 2 Matches
# MAGIC 
# MAGIC Toxicity tables Relationship Diagram
# MAGIC <div>
# MAGIC   <img src="https://cme-solution-accelerators-images.s3-us-west-2.amazonaws.com/toxicity/toxicity-erd.png"; width="40%" />
# MAGIC <div>

# COMMAND ----------

# MAGIC %md
# MAGIC ##### 4.1: Region Analysis
# MAGIC Top 5 Regions based on the number of toxic messages

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT region,
# MAGIC   Round(count(distinct account_id)) `# of toxic players`,
# MAGIC   Round(count(comment_text)) `# of toxic messages`
# MAGIC FROM Toxicity_chat_pred
# MAGIC JOIN Toxicity_players
# MAGIC ON Toxicity_chat_pred.match_id = Toxicity_players.match_id
# MAGIC JOIN Toxicity_match
# MAGIC ON Toxicity_match.match_id = Toxicity_players.match_id
# MAGIC JOIN Toxicity_cluster_regions
# MAGIC ON Toxicity_match.cluster = Toxicity_cluster_regions.cluster
# MAGIC WHERE array_contains(Toxicity_chat_pred.predicted, 'toxic')
# MAGIC GROUP BY region
# MAGIC ORDER BY count(account_id) desc, count(account_id) desc
# MAGIC LIMIT 5

# COMMAND ----------

# MAGIC %md
# MAGIC ##### 4.2: Message Analysis
# MAGIC Number of messages per label/type of toxicity

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT 'Toxic', count(*) AS Message_Count FROM Toxicity_chat_pred WHERE array_contains(Toxicity_chat_pred.predicted, 'toxic')
# MAGIC UNION
# MAGIC SELECT 'Non-Toxic', count(*) FROM Toxicity_chat_pred WHERE size(predicted) > 0

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT Toxic AS Label_Type, Message_Count from (
# MAGIC   SELECT 'Toxic', count(*) AS Message_Count FROM Toxicity_chat_pred WHERE array_contains(Toxicity_chat_pred.predicted, 'toxic')
# MAGIC   UNION
# MAGIC   SELECT 'Obscene', count(*) AS Message_Count FROM Toxicity_chat_pred WHERE array_contains(Toxicity_chat_pred.predicted, 'obscene')
# MAGIC   UNION
# MAGIC   SELECT 'Insult', count(*) AS Message_Count FROM Toxicity_chat_pred WHERE array_contains(Toxicity_chat_pred.predicted, 'insult')
# MAGIC   UNION
# MAGIC   SELECT 'Threat', count(*) AS Message_Count FROM Toxicity_chat_pred WHERE array_contains(Toxicity_chat_pred.predicted, 'threat')
# MAGIC   UNION
# MAGIC   SELECT 'Identity_hate', count(*) AS Message_Count FROM Toxicity_chat_pred WHERE array_contains(Toxicity_chat_pred.predicted, 'identity_hate')
# MAGIC   UNION
# MAGIC   SELECT 'Severe_toxic', count(*) AS Message_Count FROM Toxicity_chat_pred WHERE array_contains(Toxicity_chat_pred.predicted, 'severe_toxic')
# MAGIC )

# COMMAND ----------

# MAGIC %md
# MAGIC Number of messages per 1,2,3,4,5 labels

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT size(predicted) AS Number_of_Labels, count(*) AS Message_Count FROM Toxicity_chat_pred WHERE size(predicted) > 0 GROUP BY size(predicted) ORDER BY size(predicted) ASC

# COMMAND ----------

# MAGIC %md
# MAGIC ##### 4.3: Match Analysis
# MAGIC We can see of the 50k match dataset, 58% of the matches contained some form of toxicity. Below is the % per label. Keep in mind the Toxic label is included with all other labels.

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT Toxic AS Label_Type, Match_Count, Round((Match_Count/(SELECT count(DISTINCT match_id) AS Match_Count FROM Toxicity_chat_pred))*100) AS Percent_of_total_matches from (
# MAGIC   SELECT 'Toxic', count(DISTINCT match_id) AS Match_Count FROM Toxicity_chat_pred WHERE array_contains(Toxicity_chat_pred.predicted, 'toxic')
# MAGIC   UNION
# MAGIC   SELECT 'Obscene', count(DISTINCT match_id) AS Match_Count FROM Toxicity_chat_pred WHERE array_contains(Toxicity_chat_pred.predicted, 'obscene')
# MAGIC   UNION
# MAGIC   SELECT 'Insult', count(DISTINCT match_id) AS Match_Count FROM Toxicity_chat_pred WHERE array_contains(Toxicity_chat_pred.predicted, 'insult')
# MAGIC   UNION
# MAGIC   SELECT 'Threat', count(DISTINCT match_id) AS Match_Count FROM Toxicity_chat_pred WHERE array_contains(Toxicity_chat_pred.predicted, 'threat')
# MAGIC   UNION
# MAGIC   SELECT 'Identity_hate', count(DISTINCT match_id) AS Match_Count FROM Toxicity_chat_pred WHERE array_contains(Toxicity_chat_pred.predicted, 'identity_hate')
# MAGIC   UNION
# MAGIC   SELECT 'Severe_toxic', count(DISTINCT match_id) AS Match_Count FROM Toxicity_chat_pred WHERE array_contains(Toxicity_chat_pred.predicted, 'severe_toxic')
# MAGIC )

# COMMAND ----------

# MAGIC %md
# MAGIC Number of Toxic Messages based on match time ranges

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT Game_Lobby AS Timeline,
# MAGIC   Number_of_toxic_messages
# MAGIC   FROM (
# MAGIC     SELECT 'Game_Lobby', count(comment_text) `Number_of_toxic_messages` FROM Toxicity_chat_pred WHERE array_contains(Toxicity_chat_pred.predicted, 'toxic') AND time < 0
# MAGIC     UNION 
# MAGIC     SELECT 'Less_Than_5_minutes', count(comment_text) `Number_of_toxic_messages` FROM Toxicity_chat_pred WHERE array_contains(Toxicity_chat_pred.predicted, 'toxic') AND time < 300
# MAGIC     UNION 
# MAGIC     SELECT '5-10_minutes', count(comment_text) `Number_of_toxic_messages` FROM Toxicity_chat_pred WHERE array_contains(Toxicity_chat_pred.predicted, 'toxic') AND time BETWEEN 300 AND  600
# MAGIC     UNION
# MAGIC     SELECT '10-20_minutes', count(comment_text) `Number_of_toxic_messages` FROM Toxicity_chat_pred WHERE array_contains(Toxicity_chat_pred.predicted, 'toxic') AND time BETWEEN 600 AND  1200
# MAGIC     UNION
# MAGIC     SELECT '20-30_minutes', count(comment_text) `Number_of_toxic_messages` FROM Toxicity_chat_pred WHERE array_contains(Toxicity_chat_pred.predicted, 'toxic') AND time BETWEEN 1200 AND 1800
# MAGIC     UNION
# MAGIC     SELECT '30-40_minutes', count(comment_text) `Number_of_toxic_messages` FROM Toxicity_chat_pred WHERE array_contains(Toxicity_chat_pred.predicted, 'toxic') AND time BETWEEN 1800 AND 2400
# MAGIC     UNION
# MAGIC     SELECT '40+minutes', count(comment_text) `Number_of_toxic_messages` FROM Toxicity_chat_pred WHERE array_contains(Toxicity_chat_pred.predicted, 'toxic') AND time > 2400
# MAGIC )

# COMMAND ----------

# MAGIC %md
# MAGIC ##### 4.4: Player Analysis
# MAGIC Top 10 Players with the highest count of toxic messages

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT  account_id,
# MAGIC   count(comment_text) `# of messages`
# MAGIC FROM Toxicity_chat_pred
# MAGIC JOIN Toxicity_players
# MAGIC ON Toxicity_chat_pred.match_id = Toxicity_players.match_id
# MAGIC AND Toxicity_chat_pred.slot = Toxicity_players.player_slot
# MAGIC WHERE array_contains(Toxicity_chat_pred.predicted, 'toxic')
# MAGIC GROUP BY account_id
# MAGIC ORDER BY count(comment_text) desc
# MAGIC LIMIT 10;

# COMMAND ----------

# MAGIC %md
# MAGIC Any of these queries we could now save as our gold layer tables for consumption by the business or analysts
# MAGIC 
# MAGIC <div><img src="https://cme-solution-accelerators-images.s3-us-west-2.amazonaws.com/toxicity/delta-lake-gold.png"; width="50%" /></div>

# COMMAND ----------

# MAGIC %md
# MAGIC Copyright Databricks, Inc. [2022]. The source in this notebook is provided subject to the [Databricks License](https://databricks.com/db-license-source).  All included or referenced third party libraries are subject to the licenses set forth below.
# MAGIC 
# MAGIC |Library Name|Library license | Library License URL | Library Source URL |
# MAGIC |---|---|---|---|
# MAGIC |Spark-nlp|Apache-2.0 License| https://nlp.johnsnowlabs.com/license.html | https://www.johnsnowlabs.com/
# MAGIC |Kaggle|Apache-2.0 License |https://github.com/Kaggle/kaggle-api/blob/master/LICENSE|https://github.com/Kaggle/kaggle-api|
# MAGIC |Python|Python Software Foundation (PSF) |https://github.com/python/cpython/blob/master/LICENSE|https://github.com/python/cpython|
# MAGIC |Spark|Apache-2.0 License |https://github.com/apache/spark/blob/master/LICENSE|https://github.com/apache/spark|
