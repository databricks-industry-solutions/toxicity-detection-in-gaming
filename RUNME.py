# Databricks notebook source
# MAGIC %md This notebook sets up the companion cluster(s) to run the solution accelerator. It also creates the Workflow to create a Workflow DAG and illustrate the order of execution. Feel free to interactively run notebooks with the cluster or to run the Workflow to see how this solution accelerator executes. Happy exploring!
# MAGIC 
# MAGIC The pipelines, workflows and clusters created in this script are not user-specific, so if any user alters the workflow and cluster via UI, running this script resets the changes.
# MAGIC 
# MAGIC **Note**: If the job execution fails, please confirm that you have set up other environment dependencies as specified in the accelerator notebooks. Accelerators sometimes require the user to set up additional cloud infra or data access, for instance. 

# COMMAND ----------

# DBTITLE 0,Install util packages
# MAGIC %pip install git+https://github.com/databricks-academy/dbacademy-rest git+https://github.com/databricks-academy/dbacademy-gems git+https://github.com/databricks-industry-solutions/notebook-solution-companion

# COMMAND ----------

from solacc.companion import NotebookSolutionCompanion

# COMMAND ----------

job_json = {
        "timeout_seconds": 36000,
        "max_concurrent_runs": 1,
        "tags": {
            "usage": "solacc_testing",
            "group": "CME"
        },
        "tasks": [
            {
                "job_cluster_key": "gaming_cluster",
                "notebook_task": {
                    "notebook_path": f"00_context",
                    "base_parameters": {
                        "env": "test"
                    }
                },
                "task_key": "Gaming_00"
            },
            {
                "job_cluster_key": "gaming_cluster",
                "notebook_task": {
                    "notebook_path": f"01_intro",
                    "base_parameters": {
                        "env": "test"
                    }
                },
                "task_key": "Gaming_01",
                "depends_on": [
                    {
                        "task_key": "Gaming_00"
                    }
                ]
            },
            {
                "job_cluster_key": "gaming_cluster",
                "libraries": [],
                "notebook_task": {
                    "notebook_path": f"02_load_data"
                },
                "task_key": "Gaming_02",
                "depends_on": [
                    {
                        "task_key": "Gaming_01"
                    }
                ]
            },
            {
                "job_cluster_key": "gaming_cluster",
                "notebook_task": {
                    "notebook_path": f"03_simple_classification"
                },
                "libraries": [
                    {
                        "maven": {
                            "coordinates": "com.johnsnowlabs.nlp:spark-nlp_2.12:4.0.0"
                        }
                    }
                ],
                "task_key": "Gaming_03",
                "depends_on": [
                    {
                        "task_key": "Gaming_02"
                    }
                ]
            },
            {
                "job_cluster_key": "gaming_cluster",
                "notebook_task": {
                    "notebook_path": f"04_inference_eda"
                },
                "libraries": [
                    {
                        "maven": {
                            "coordinates": "com.johnsnowlabs.nlp:spark-nlp_2.12:4.0.0"
                        }
                    }
                ],
                "task_key": "Gaming_04",
                "depends_on": [
                    {
                        "task_key": "Gaming_03"
                    }
                ]
            }
        ],
        "job_clusters": [
            {
                "job_cluster_key": "gaming_cluster",
                "new_cluster": {
                    "spark_version": "10.4.x-cpu-ml-scala2.12",
                "spark_conf": {
                    "spark.databricks.delta.formatCheck.enabled": "false"
                    },
                    "num_workers": 4,
                    "node_type_id": {"AWS": "i3.xlarge", "MSA": "Standard_D3_v2", "GCP": "n1-highmem-4"}, # different from standard API,
                    "custom_tags": {
                        "usage": "solacc_testing"
                    },
                    "spark_conf": {
                        "spark.serializer": "org.apache.spark.serializer.KryoSerializer",
                        "spark.kryoserializer.buffer.max": "2000M"
                    },
                }
            }
        ]
    }

# COMMAND ----------

NotebookSolutionCompanion().deploy_compute(job_json)

# COMMAND ----------



# COMMAND ----------


