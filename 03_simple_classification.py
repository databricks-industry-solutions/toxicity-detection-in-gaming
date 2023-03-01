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
# MAGIC ### In this notebook you:
# MAGIC * Configure the environment
# MAGIC * Explore the training data
# MAGIC * Prep the training data
# MAGIC * Understand sentence embeddings
# MAGIC * Build embedding and classification pipelines
# MAGIC * Track model training with MLflow
# MAGIC   * Train
# MAGIC   * Tune
# MAGIC   * Evaluate
# MAGIC   * Register

# COMMAND ----------

# MAGIC %md
# MAGIC ##Step 1: Configure the Environment
# MAGIC * To use this notebook, the cluster must be configured to support Spark NLP. We provide a RUNME notebook to automate the bulk of the configuration process. 
# MAGIC * If you run this notebook as part of the automated Workflow created in the `RUNME` notebook, there is no additional configuration needed. If you run this notebook interactively, please add the following library to the `gaming_cluster` created in `RUNME`.
# MAGIC   * **Install libraries:**
# MAGIC     * Maven Coordinates:
# MAGIC       * CPU: `com.johnsnowlabs.nlp:spark-nlp_2.12:4.0.0`
# MAGIC       * GPU: `com.johnsnowlabs.nlp:spark-nlp-gpu_2.12:4.0.0`
# MAGIC     
# MAGIC   * **A note on cluster instances**: A CPU or GPU cluster can be used to run this notebook. Feel free to edit the `gaming_cluster` and explore using a GPU cluster for potential speedup.

# COMMAND ----------

# MAGIC %run ./config/notebook_config

# COMMAND ----------

import sparknlp
from sparknlp.annotator import *
from sparknlp.common import *
from sparknlp.base import *

from pyspark.ml import Pipeline
from pyspark.mllib.evaluation import MultilabelMetrics
from pyspark.sql.functions import lit,when,col,array,array_contains,array_remove,regexp_replace,size,when
from pyspark.sql.types import ArrayType,DoubleType,StringType

from pyspark.ml.evaluation import MultilabelClassificationEvaluator

import mlflow
import mlflow.spark
from mlflow.tracking import MlflowClient

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 2: Explore the Training Dataset
# MAGIC 
# MAGIC The training dataset from Jigsaw contains 6 columns that denote what labels are associated with a given comment. This is denoted by a 0 for false and a 1 for true.

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT * FROM Toxicity_training WHERE toxic = 0 LIMIT 1

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 3: Training Data Prep

# COMMAND ----------

dataPrepDF = spark.table("Toxicity_training")\
  .withColumnRenamed("toxic","toxic_true")\
  .withColumnRenamed("severe_toxic","severe_toxic_true")\
  .withColumnRenamed("obscene","obscene_true")\
  .withColumnRenamed("threat","threat_true")\
  .withColumnRenamed("insult","insult_true")\
  .withColumnRenamed("identity_hate","identity_hate_true")\
  .withColumn('toxic',when(col('toxic_true') == '1','toxic').otherwise(0))\
  .withColumn('severe_toxic',when(col('severe_toxic_true') == '1','severe_toxic').otherwise(0))\
  .withColumn('obscene',when(col('obscene_true') == '1','obscene').otherwise(0))\
  .withColumn('threat',when(col('threat_true') == '1','threat').otherwise(0))\
  .withColumn('insult',when(col('insult_true') == '1','insult').otherwise(0))\
  .withColumn('identity_hate',when(col('identity_hate_true') == '1','identity_hate').otherwise(0))\
  .withColumn('labels',array_remove(array('toxic','severe_toxic','obscene','threat','insult','identity_hate'),'0')\
              .astype(ArrayType(StringType())))\
  .drop('toxic','severe_toxic','obscene','threat','insult','identity_hate')\
  .withColumn('label_true', array(
    col('toxic_true').cast(DoubleType()),
    col('severe_toxic_true').cast(DoubleType()),
    col('obscene_true').cast(DoubleType()),
    col('threat_true').cast(DoubleType()),
    col('insult_true').cast(DoubleType()),
    col('identity_hate_true').cast(DoubleType()))
  )
  
train, val = dataPrepDF.randomSplit([0.8,0.2],42)

# COMMAND ----------

# MAGIC %md
# MAGIC ##### 3.1: Display training data

# COMMAND ----------

display(train.limit(1).filter(size(col('labels')) == 0))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 4: Model Pipeline
# MAGIC 
# MAGIC * The models built in this notebook uses Spark NLP to classify toxic comments and is an adaptation of a [demo notebook](https://nlp.johnsnowlabs.com/2021/01/21/multiclassifierdl_use_toxic_sm_en.html) published by John Snow Labs.
# MAGIC * Spark NLP is an open source library that is built on top of Apache Spark&trade; and Spark ML. A few benefits of using Spark NLP include:
# MAGIC   * `Start-of-the-Art:` pre-trained algorithms available out-of-the-box
# MAGIC   * `Efficient:` single processing framework mitigates serializing/deserializing overhead
# MAGIC   * `Enterprise Ready:` successfully deployed by many large enterprises.
# MAGIC   
# MAGIC * Further information on Spark-NLP and more can be found [here](https://towardsdatascience.com/introduction-to-spark-nlp-foundations-and-basic-components-part-i-c83b7629ed59).
# MAGIC   * [Transformers documentation](https://nlp.johnsnowlabs.com/docs/en/transformers) used in pipeline
# MAGIC   * [Annotators documentation](https://nlp.johnsnowlabs.com/docs/en/annotators) used in pipeline
# MAGIC 
# MAGIC Lets jump in and build our pipeline. 
# MAGIC   * [Document Assembler](https://nlp.johnsnowlabs.com/docs/en/transformers#documentassembler-getting-data-in) creates the first annotation of type Document from the contents of our dataframe. This is used by the annotators in subsequent steps.
# MAGIC   * Embeddings map words to vectors. A great explanation on this topic can be found [here](https://towardsdatascience.com/word-embeddings-exploration-explanation-and-exploitation-with-code-in-python-5dac99d5d795). The embeddings serve as an input for our classifier.
# MAGIC   
# MAGIC <div><img src="https://cme-solution-accelerators-images.s3-us-west-2.amazonaws.com/toxicity/nlp_pipeline.png"; width="70%"></div>

# COMMAND ----------

# MAGIC %md
# MAGIC ##### 4.1: Define the Document Assembler and Embedding Stages
# MAGIC 
# MAGIC Per the [documentation from John Snow Labs on Universal Sentence Embeddings](https://nlp.johnsnowlabs.com/2020/04/17/tfhub_use.html): 
# MAGIC * "The model is trained and optimized for greater-than-word length text, such as sentences, phrases or short paragraphs. It is trained on a variety of data sources and a variety of tasks with the aim of dynamically accommodating a wide variety of natural language understanding tasks. The input is variable length English text and the output is a 512 dimensional vector. We apply this model to the STS benchmark for semantic similarity, and the results can be seen in the example notebook made available. The universal-sentence-encoder model is trained with a deep averaging network (DAN) encoder.""

# COMMAND ----------

document_assembler = DocumentAssembler() \
  .setInputCol("comment_text") \
  .setOutputCol("document")

universal_embeddings = UniversalSentenceEncoder.pretrained() \
  .setInputCols(["document"]) \
  .setOutputCol("universal_embeddings")  

# COMMAND ----------

# MAGIC %md
# MAGIC ##### 4.2: Define the Classifier Stage
# MAGIC [MultiClassifier DL Approach](https://nlp.johnsnowlabs.com/docs/en/annotators#multiclassifierdl-multi-label-text-classification) is a Multi-label Text Classification. MultiClassifierDL uses a Bidirectional GRU with Convolution model that was built inside TensorFlow.

# COMMAND ----------

threshold = 0.7
batchSize = 32
maxEpochs = 10
learningRate = 1e-3

ClassifierDL = MultiClassifierDLApproach() \
  .setInputCols(["universal_embeddings"]) \
  .setOutputCol("class") \
  .setLabelColumn("labels") \
  .setMaxEpochs(maxEpochs) \
  .setLr(learningRate) \
  .setBatchSize(batchSize) \
  .setThreshold(threshold) \
  .setOutputLogsPath('./') \
  .setEnableOutputLogs(False)

# COMMAND ----------

# MAGIC %md
# MAGIC ##### 4.3: Define the model pipeline
# MAGIC Here we add the 3 stages into a pipeline for use during training. This pipeline will be used in step 5 below.

# COMMAND ----------

EndToEndPipeline = Pipeline(stages=[
  document_assembler,
  universal_embeddings,
  ClassifierDL
])

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 5: Multilabel Classification Training
# MAGIC 
# MAGIC Steps in the classification workflow
# MAGIC * Create experiment and enable [autologging for spark](https://www.mlflow.org/docs/latest/python_api/mlflow.spark.html).
# MAGIC * Train and evaluate the model, logging model metrics along the way.
# MAGIC * End the tracked run. Results will be viewable in the experiments tab on the top right of the UI.
# MAGIC 
# MAGIC 
# MAGIC <div><img src="https://cme-solution-accelerators-images.s3-us-west-2.amazonaws.com/toxicity/mlflow.png"; width="60%"></div>

# COMMAND ----------

# MAGIC %md
# MAGIC ##### 5.1: Create Experiment & Start Autologging
# MAGIC 
# MAGIC We want experiments to persist outside of this notebook and to allow others to collaborate with their work on the same project.
# MAGIC * Create experiment in users folder to hold model artifacts and parameters
# MAGIC 
# MAGIC Note: When running this code for production, change the experiment path to a location outside of a user's personal folder.

# COMMAND ----------

# Create an experiment with a name that is unique and case sensitive.
client = MlflowClient()

mlflow.set_experiment(f"/Users/{dbutils.notebook.entry_point.getDbutils().notebook().getContext().tags().apply('user')}/Toxicity_Classification")

mlflow.spark.autolog()

# COMMAND ----------

# MAGIC %md
# MAGIC Under your user folder, you will find an experiment created to log the runs with parameters and metrics. The model will also be logged in the model registry during the run, similar to the image on the right.
# MAGIC <div><img src="https://cme-solution-accelerators-images.s3-us-west-2.amazonaws.com/toxicity/mlflow-experiments.png"; width="55%">     ->     <img src="https://cme-solution-accelerators-images.s3-us-west-2.amazonaws.com/toxicity/mlflow-model-registry.png"; width="40%"></div>

# COMMAND ----------

# MAGIC %md
# MAGIC #### 5.2: Train and Evaluate the Model
# MAGIC 
# MAGIC Note: This implementation uses the basic fit & transform without cross validation or parameter grids.

# COMMAND ----------

with mlflow.start_run():
  
  mlflow.log_param('threshold',threshold)
  mlflow.log_param('batchSize',batchSize)
  mlflow.log_param('maxEpochs',maxEpochs)
  mlflow.log_param('learningRate',learningRate)
  
  model = EndToEndPipeline.fit(train)
  
  mlflow.spark.log_model(model,"spark-model",registered_model_name='Toxicity MultiLabel Classification', pip_requirements=["spark-nlp"])
  
  #supports "f1" (default), "weightedPrecision", "weightedRecall", "accuracy"
  evaluator = MultilabelClassificationEvaluator(labelCol="label_true",predictionCol="label_pred")
  
  predictions = model.transform(val)\
   .withColumn('label_pred', array(
       when(array_contains(col('class.result'),'toxic'),1).otherwise(0).cast(DoubleType()),
       when(array_contains(col('class.result'),'severe_toxic'),1).otherwise(0).cast(DoubleType()),
       when(array_contains(col('class.result'),'obscene'),1).otherwise(0).cast(DoubleType()),
       when(array_contains(col('class.result'),'threat'),1).otherwise(0).cast(DoubleType()),
       when(array_contains(col('class.result'),'insult'),1).otherwise(0).cast(DoubleType()),
       when(array_contains(col('class.result'),'identity_hate'),1).otherwise(0).cast(DoubleType())
     )
   )
  
  score = evaluator.evaluate(predictions)
  mlflow.log_metric('f1',score)
  print(score)
  
mlflow.end_run()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Next Steps
# MAGIC 
# MAGIC * Build and run advanced classification pipeline

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
