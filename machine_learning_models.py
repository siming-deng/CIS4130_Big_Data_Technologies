# install packages
pip3 install pandas
pip3 install numpy
pip3 install matplotlib
pip3 install seaborn
pip3 install boto3
pip3 install io
pip3 install s3fs

# start the pyspark environment in aws emr cluster
pyspark

sc.setLogLevel("ERROR")

# import necessary packages
import io
import pandas as pd
import numpy as np
import boto3
import matplotlib.pyplot as plt
import seaborn as sns
import s3fs

from pyspark.sql import functions as F
from pyspark.sql.functions import col,isnan, when, count, udf
from pyspark.ml.feature import StringIndexer, OneHotEncoder, Binarizer
from pyspark.ml.feature import Tokenizer, RegexTokenizer, StopWordsRemover, VectorAssembler
from nltk.stem.snowball import SnowballStemmer
from pyspark.sql.types import StringType, ArrayType, DoubleType
from pyspark.ml.feature import HashingTF, IDF
from pyspark.ml import Pipeline

from pyspark.ml.classification import LogisticRegression, LogisticRegressionModel
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.evaluation import BinaryClassificationEvaluator, BinaryClassificationMetrics

transformed_df = spark.read.parquet('s3://my-data-bucket-sd/transformed_amazon_reviews_us_ver4.parquet/')

assembler = VectorAssembler(inputCols = ['product_category_vector', 'verified_purchase_vector', \
	'helpful_votes_vector', 'wordCount_vector', 'features'], outputCol = 'combined_features')

aws_assembled = assembler.transform(transformed_df)

aws_assembled = aws_assembled.drop('words_stemmed', 'product_category_vector', 'verified_purchase_vector', \
	'helpful_votes_vector', 'wordCount_vector', 'features')

aws_assembled.show()

aws_assembled = aws_assembled.withColumnRenamed('combined_features', 'features')

aws_assembled.show()

# split the data into training and test sets
training_data, test_data = aws_assembled.randomSplit([0.7, 0.3], seed = 42)

# create a logistic regression estimator
lr = LogisticRegression(maxIter = 100)

# create a grid to hold hyperparameters
grid = ParamGridBuilder()
grid = grid.addGrid(lr.regParam, [0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
grid = grid.addGrid(lr.elasticNetParam, [0.0, 0.5, 1.0])

# build the parameter grid
grid = grid.build()

print('Number of models to be tested:', len(grid))

# create a binary classification evaluator to evaluate how well model works
evaluator = BinaryClassificationEvaluator(metricName = 'areaUnderROC')

# create the cross validator using the hyperparameter grid
cv = CrossValidator(estimator = lr, 
	estimatorParamMaps = grid, 
	evaluator = evaluator, 
	numFolds = 3,
	seed = 42)

# train the model
cv = cv.fit(training_data)
cv.avgMetrics

# test the predictions
predictions = cv.transform(test_data)

# calculate auc
auc = evaluator.evaluate(predictions)
print('AUC:', auc)

# create confusion matrix
predictions.groupby('label').pivot('prediction').count().fillna(0).show()
cm = predictions.groupby('label').pivot('prediction').count().fillna(0).collect()

def calculate_precision_recall(cm):
	tn = cm[1][1]
	fp = cm[1][2]
	fn = cm[0][1]
	tp = cm[0][2]
	precision = tp / (tp + fp)
	recall = tp / (tp + fn)
	accuracy = (tp + tn) / (tp + tn + fp + fn)
	f1_score = 2 * ((precision * recall) / (precision + recall))
	return accuracy, precision, recall, f1_score

print("The model accuracy is:", calculate_precision_recall(cm)[0])
print("The model precision is:", calculate_precision_recall(cm)[1])
print("The model recall is:", calculate_precision_recall(cm)[2])
print("The model f1_score is:", calculate_precision_recall(cm)[3])

parammap = cv.bestModel.extractParamMap()

for p, v in parammap.items():
	print(p, v)

# grab the best model
mymodel = cv.bestModel

plt.figure(figsize = (15,8))
plt.plot([0,1], [0,1], 'r--')
x = mymodel.summary.roc.select('FPR').collect()
y = mymodel.summary.roc.select('TPR').collect()
plt.scatter(x, y)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title("ROC Curve")

# create a buffer to hold the figure
img_data = io.BytesIO()
plt.savefig(img_data, format = "png", bbox_inches='tight')
img_data.seek(0)

# connect to the s3fs file system
s3 = s3fs.S3FileSystem(anon = False)
with s3.open('s3://my-data-bucket-sd/review_roc_curve.png', 'wb') as f:
	f.write(img_data.getbuffer())








