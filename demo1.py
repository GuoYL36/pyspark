from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.ml.feature import VectorAssembler
import sparkxgb
from sparkxgb import XGBoostEstimator

import os
os.environ['PYSPARK_SUBMIT_ARGS'] = '--jars xgboost4j-spark-0.72.jar,xgboost4j-0.72.jar pyspark-shell'  # jar包放在当前代码的路径下



if __name__ == "__main__":
    spark = SparkSession \
        .builder \
        .master("local") \
        .appName("PythonWordCount") \
        .getOrCreate()

    spark.sparkContext.addPyFile("./sparkxgb.zip")

    data = spark.read.option("header","true").csv("./data.csv")
    data.show(10)

    data1 = data.withColumn("label", data.fragment_id>50)
    data1 = data1.withColumn("label", data1.label.cast("Double"))

    data1.show(10)
    cols = ["x","y","z"]
    for i in cols:
        data1 = data1.withColumn(i, data1[i].cast("Double"))

    vecAssembler = VectorAssembler(inputCols=cols,outputCol="features")
    data2 = vecAssembler.transform(data1)


    data2.show(10)

    trainDF, testDF = data2.randomSplit([0.8,0.2],seed=24)


    xgboost = XGBoostEstimator(featuresCol="features",
                                   labelCol="label",
                                   predictionCol="prediction",
                                   missing=float("+inf"))

    model = xgboost.fit(trainDF)

    pred = model.transform(testDF)
    pred.show()
    spark.stop()
