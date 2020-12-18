from sparkxgb import XGBoostEstimator
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
import os
os.environ['PYSPARK_SUBMIT_ARGS'] = '--jars xgboost4j-spark-0.72.jar,xgboost4j-0.72.jar pyspark-shell'    # jar包放在当前代码的路径下


spark = SparkSession \
    .builder \
    .master("local") \
    .appName("PythonWordCount") \
    .getOrCreate()
spark.sparkContext.addPyFile("./sparkxgb.zip")  # zip包路径

# Load Data
dataPath = "sample_binary_classification_data.txt"
dataDF = spark.read.format("libsvm").load(dataPath)

# Split into Train/Test
trainDF, testDF = dataDF.randomSplit([0.8, 0.2], seed=1000)

# Define and train model
xgboost = XGBoostEstimator(
    # General Params
    nworkers=1, nthread=1, checkpointInterval=-1, checkpoint_path="",
    use_external_memory=False, silent=0, missing=float("nan"),

    # Column Params
    featuresCol="features", labelCol="label", predictionCol="prediction",
    weightCol="weight", baseMarginCol="baseMargin",

    # Booster Params
    booster="gbtree", base_score=0.5, objective="binary:logistic", eval_metric="error",
    num_class=2, num_round=2, seed=None,

    # Tree Booster Params
    eta=0.3, gamma=0.0, max_depth=6, min_child_weight=1.0, max_delta_step=0.0, subsample=1.0,
    colsample_bytree=1.0, colsample_bylevel=1.0, reg_lambda=0.0, alpha=0.0, tree_method="auto",
    sketch_eps=0.03, scale_pos_weight=1.0, grow_policy='depthwise', max_bin=256,

    # Dart Booster Params
    sample_type="uniform", normalize_type="tree", rate_drop=0.0, skip_drop=0.0,

    # Linear Booster Params
    lambda_bias=0.0
)
xgboost_model = xgboost.fit(trainDF)

# Transform test set
xgboost_model.transform(testDF).show()

# Write model/classifier
xgboost.write().overwrite().save("xgboost_class_test")
xgboost_model.write().overwrite().save("xgboost_class_test.model")