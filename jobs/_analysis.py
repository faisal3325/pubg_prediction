"""Run a linear regression using Apache Spark ML.

In the following PySpark (Spark Python API) code, we take the following actions:

  * Load a previously created linear regression (BigQuery) input table
    into our Cloud Dataproc Spark cluster as an RDD (Resilient
    Distributed Dataset)
  * Transform the RDD into a Spark Dataframe
  * Vectorize the features on which the model will be trained
  * Compute a linear regression using Spark ML

"""
from pyspark import SparkContext, SparkConf
from pyspark.sql import SparkSession
from pyspark.sql.types import FloatType
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.evaluation import RegressionEvaluator


from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

import logging
logging.basicConfig(level=logging.INFO)

def calculate_error(cl,name):
  logging.info(name)
  logging.info('Mean Absolute Error is {:.5f}'.format(mean_absolute_error(y_val, cl.predict(X_val))))
  logging.info('R2 score is {:.2%}'.format(r2_score(y_val, cl.predict(X_val))))

spark = (SparkSession.builder.appName('bigquery').getOrCreate())
logging.info('fetching data')
# Read the data from BigQuery as a Spark Dataframe.
train_data = spark.read.format("bigquery").option("credentialsFile", "credentials.json").option("project", "lyit-260817").option("parentProject", "lyit-260817").option("table", "lyit-260817.pubg.train_processed").load()

y = spark.read.format("bigquery").option("credentialsFile", "credentials.json").option("project", "lyit-260817").option("parentProject", "lyit-260817").option("table", "lyit-260817.pubg.train_processed_y").load()
logging.info('fetched data')

cols = train_data.columns
print(cols)

for col in cols:
  train_data = train_data.withColumn(col, train_data[col].cast(FloatType()))

#Splitting the data into test and train
X_train, X_val, y_train, y_val = train_test_split(train_data.toPandas(), y.toPandas(), train_size=0.7)
linear = LinearRegression(copy_X=True)
linear.fit(X_train,y_train)
calculate_error(linear,"linear")

ridge = Ridge(copy_X=True)
ridge.fit(X_train,y_train)
calculate_error(ridge,"ridge")

lasso = Lasso(copy_X=True)
lasso.fit(X_train,y_train)
calculate_error(lasso,"lasso")

elastic = ElasticNet(copy_X=True)
elastic.fit(X_train,y_train)
calculate_error(elastic,"elastic")

ada = AdaBoostRegressor(learning_rate=0.8)
ada.fit(X_train,y_train)
calculate_error(ada,"Adaboost")

GBR = GradientBoostingRegressor(learning_rate=0.8)
GBR.fit(X_train,y_train)
calculate_error(GBR,"GBR")

forest = RandomForestRegressor(n_estimators=10)
forest.fit(X_train,y_train)
calculate_error(forest,"forest")

tree = DecisionTreeRegressor()
tree.fit(X_train,y_train)
calculate_error(tree,"tree")