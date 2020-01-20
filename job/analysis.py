"""Run a linear regression using Apache Spark ML.

In the following PySpark (Spark Python API) code, we take the following actions:

  * Load a previously created linear regression (BigQuery) input table
    into our Cloud Dataproc Spark cluster as an RDD (Resilient
    Distributed Dataset)
  * Transform the RDD into a Spark Dataframe
  * Vectorize the features on which the model will be trained
  * Compute a linear regression using Spark ML

"""

from __future__ import print_function
from pyspark.context import SparkContext
from pyspark.ml.linalg import Vectors
from pyspark.ml.regression import LinearRegression
from pyspark.ml.feature import StringIndexer
from pyspark.sql.session import SparkSession
# The imports, above, allow us to access SparkML features specific to linear
# regression as well as the Vectors types.

# Helper functions
# to get all categorical variables
def getCatVariables(df):
  col = df.dtypes
  categorical_variables = []
  # print(col)
  for k, v in col:
    # print(k, v)
    if v == 'string':
      categorical_variables.append(k)

  return categorical_variables

# Label encoding a variable
def labelEncode(df, df_col):
  indexer = StringIndexer(inputCol=df_col, outputCol=f'encoded_{df_col}')
  indexed = indexer.fit(df).transform(df)
  return indexed

# Define a function that collects the features of interest
# Package the vector in a tuple containing the label for that row.
def vector_from_inputs(r):
  print("r", r)
  
  print("FAISAL", r["winPlacePerc"], Vectors.dense(float(r["assists"]), float(r["boosts"]), float(r["damageDealt"]), float(r["DBNOs"]), float(r["headshotKills"]), float(r["heals"]), float(r["killPlace"]), float(r["killPoints"]), float(r["kills"]), float(r["killStreaks"]), float(r["longestKill"]), float(r["matchDuration"]), float(r["maxPlace"]), float(r["numGroups"]), float(r["rankPoints"]), float(r["revives"]), float(r["rideDistance"]), float(r["roadKills"]), float(r["swimDistance"]), float(r["teamKills"]), float(r["vehicleDestroys"]), float(r["walkDistance"]), float(r["weaponsAcquired"]), float(r["winPoints"]), float(r["encoded_matchType"])))

  return (r["winPlacePerc"], Vectors.dense(float(r["assists"]), float(r["boosts"]), float(r["damageDealt"]), float(r["DBNOs"]), float(r["headshotKills"]), float(r["heals"]), float(r["killPlace"]), float(r["killPoints"]), float(r["kills"]), float(r["killStreaks"]), float(r["longestKill"]), float(r["matchDuration"]), float(r["maxPlace"]), float(r["numGroups"]), float(r["rankPoints"]), float(r["revives"]), float(r["rideDistance"]), float(r["roadKills"]), float(r["swimDistance"]), float(r["teamKills"]), float(r["vehicleDestroys"]), float(r["walkDistance"]), float(r["weaponsAcquired"]), float(r["winPoints"]), float(r["encoded_matchType"])))

# assists, boosts, damageDealt, DBNOs, headshotKills, heals, killPlace, killPoints, kills, killStreaks, longestKill, matchDuration, maxPlace, numGroups, rankPoints, revives, rideDistance, roadKills, swimDistance, teamKills, vehicleDestroys, walkDistance, weaponsAcquired, winPoints, winPlacePerc, encoded_matchType

def get_dummy(df, indexCol, categoricalCols, continuousCols, labelCol):
  from pyspark.ml import Pipeline
  from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler
  from pyspark.sql.functions import col

  indexers = [ StringIndexer(inputCol=c, outputCol="{0}_indexed".format(c)) for c in categoricalCols ]

  # default setting: dropLast=True
  encoders = [OneHotEncoder(inputCol=indexer.getOutputCol(), outputCol="{0}_encoded".format(indexer.getOutputCol())) for indexer in indexers]

  assembler = VectorAssembler(inputCols=[encoder.getOutputCol() for encoder in encoders] + continuousCols, outputCol="features")

  pipeline = Pipeline(stages=indexers + encoders + [assembler])

  model=pipeline.fit(df)
  data = model.transform(df)

  data = data.withColumn('label', col(labelCol))

  return data.select(indexCol, 'features', 'label')

spark = (SparkSession.builder.appName('bigquery').getOrCreate())

# Read the data from BigQuery as a Spark Dataframe.
train_data = spark.read.format("bigquery").option("credentialsFile", "/pubg_prediction/credentials.json").option("project", "lyit-260817").option("parentProject", "lyit-260817").option("table", "lyit-260817.pubg.train_mini").load()

# test_data = spark.read.format("bigquery").option("credentialsFile", "/pubg_prediction/credentials.json").option("project", "lyit-260817").option("parentProject", "lyit-260817").option("table", "lyit-260817.pubg.test").load()

print(train_data)
categorical_variables = getCatVariables(train_data)
print('categorical_variables', categorical_variables)

labels_to_retain = ['matchType']

col_to_drop = []

for col in categorical_variables:
  if col not in labels_to_retain:
    col_to_drop.append(col)

for col in col_to_drop:
  train_data = train_data.drop(col)
  # test_data = test_data.drop(col)

for col in labels_to_retain:
  train_data = labelEncode(train_data, col)
  train_data = train_data.drop(col)
  # test_data = test_data.drop(col)

x = 'winPlacePerc'
y = []

for col in train_data.columns:
  if x is not col:
    y.append(col)

# Create an input DataFrame for Spark ML using the above function.
training_data = train_data.rdd.map(vector_from_inputs).toDF(["label", "features"])
training_data.cache()
print(training_data)
# Construct a new LinearRegression object and fit the training data.
lr = LinearRegression(maxIter=5, regParam=0.2, solver="normal")
model = lr.fit(training_data)
# Print the model summary.
print("Coefficients:" + str(model.coefficients))
print("Intercept:" + str(model.intercept))
print("R^2:" + str(model.summary.r2))
model.summary.residuals.show()
model.summary.show()
