# pubg_prediction
Source code of PUBG Finish Placement Prediction implemented in GCP

### Running on Docker
```sh
$ docker container run -v pubg_prediction/:/pubg_prediction/ godatadriven/pyspark --jars /pubg_prediction/bigquery_spark-bigquery-latest.jar /pubg_prediction/job/analysis.py
```

You will need godatadriven/pyspark image on Docker.