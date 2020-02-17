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

```sh
$ gcloud beta dataproc clusters create cluster-43a7 --enable-component-gateway --bucket dataset_pubg --region us-central1 --subnet default --zone us-central1-b --master-machine-type n1-standard-1 --master-boot-disk-size 15 --num-workers 2 --worker-machine-type n1-standard-1 --worker-boot-disk-size 15 --image-version 1.3-deb9 --optional-components ANACONDA,JUPYTER --max-idle 600s --project lyit-260817
```