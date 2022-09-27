<div >
  <img src="https://cme-solution-accelerators-images.s3-us-west-2.amazonaws.com/toxicity/solution-accelerator-logo.png"; width="50%">
</div>


## Overview

Toxicity can have a large impact on player engagement and satisfaction. Game companies are working on ways to address forms of toxicity in their platforms. One of the most common interactions with toxicity is in chat boxes or in-game messaging systems. As companies are becoming more data driven, the opportunity to detect toxicity using the data at hand is present, but technically challenging. This solution accelerator is a head start on deploying a ML-enhanced data pipeline to address toxic messages in real time.

** Authors**
- Duncan Davis [<duncan.davis@databricks.com>]
- Dan Morris [<dan.morris@databricks.com>]
___

* This series of notebooks is intended to help you use multi-label classification to detect and analyze toxicity in your data.

* In support of this goal, we will:
 * Load toxic-comment training data from Jigsaw and game data from Dota 2.
 * Create one pipeline for streaming and batch to detect toxicity in near real-time and/or on an ad-hoc basis. This pipeline can then be used for managing tables for reporting, ad hoc queries, and/or decision support.
 * Label text chat data using Multi-Label Classification.
 * Create a dashboard for monitoring the impact of toxicity.

&copy; 2022 Databricks, Inc. All rights reserved. The source in this notebook is provided subject to the Databricks License [https://databricks.com/db-license-source].  All included or referenced third party libraries are subject to the licenses set forth below.

|Library Name|Library license | Library License URL | Library Source URL |
|---|---|---|---|
|Spark-nlp|Apache-2.0 License| https://nlp.johnsnowlabs.com/license.html | https://www.johnsnowlabs.com/
|Kaggle|Apache-2.0 License |https://github.com/Kaggle/kaggle-api/blob/master/LICENSE|https://github.com/Kaggle/kaggle-api|
|Python|Python Software Foundation (PSF) |https://github.com/python/cpython/blob/master/LICENSE|https://github.com/python/cpython|
|Spark|Apache-2.0 License |https://github.com/apache/spark/blob/master/LICENSE|https://github.com/apache/spark|

To run this accelerator, clone this repo into a Databricks workspace. Attach the RUNME notebook to any cluster running a DBR 11.0 or later Runtime, and execute the notebook via `Run-All`. A multi-step-job describing the accelerator pipeline will be created, and the link will be provided. Execute the multi-step-job to see how the pipeline runs.

The job configuration is written in the RUNME notebook in json format. The cost associated with running the accelerator is the user's responsibility.
