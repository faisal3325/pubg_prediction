# pubg_prediction
Source code of PUBG Finish Placement Prediction implemented in GCP

### Running on Docker
```sh
$ docker container run -v pubg_prediction/:/pubg_prediction/ godatadriven/pyspark --jars /pubg_prediction/bigquery_spark-bigquery-latest.jar /pubg_prediction/job/analysis.py
```

You will need godatadriven/pyspark image on Docker.

### Workflow templates
## Creating workflow Template
```sh
$ gcloud dataproc workflow-templates create pyspark-pubg --region us-central1
```

## Creating cluster
```sh
$ gcloud dataproc workflow-templates set-managed-cluster pyspark-pubg --master-machine-type n1-standard-1 --worker-machine-type n1-standard-1 --num-workers 2 --cluster-name pyspark-cluster --region us-central1
```

## Submitting job
```sh
$ gcloud dataproc workflow-templates add-job job-type --step-id baz --start-after foo,bar --workflow-template my-workflow --space separated job args
```