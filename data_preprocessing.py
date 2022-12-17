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

from pyspark.ml.classification import RandomForestClassifier, LogisticRegression
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.evaluation import BinaryClassificationEvaluator, BinaryClassificationMetrics

parquet_df = spark.read.parquet('s3a://my-data-bucket-sd/cleaned_amazon_reviews_us_ver2.parquet/')

parquet_df = parquet_df.sample(0.25)

parquet_df.count()

# look at null values in each column
parquet_df.select([count(when(isnan(c) | col(c).isNull(), c)).alias(c) for c in ["star_rating", \
	"product_category", "helpful_votes", "verified_purchase", "clean_review_body", "wordCount"]]).show()

# create a binary output label
parquet_df = parquet_df.withColumn("label", when(col("star_rating") > 3, 1.0).otherwise(0.0))

# cast the helpful votes and word count to double
parquet_df = parquet_df.withColumn("helpful_votes", parquet_df.helpful_votes.cast(DoubleType()))
parquet_df = parquet_df.withColumn("wordCount", parquet_df.wordCount.cast(DoubleType()))

parquet_df.show()

string_indexer = StringIndexer(inputCols = ['product_category', 'verified_purchase'], \
	outputCols = ['product_category_index', 'verified_purchase_index'])

aws_indexer = string_indexer.fit(parquet_df).transform(parquet_df)

aws_indexer = aws_indexer.drop("product_category", "verified_purchase")

aws_indexer.show()

encoder = OneHotEncoder(inputCols = ['product_category_index', 'verified_purchase_index', 'helpful_votes', 'wordCount'], \
	outputCols = ['product_category_vector', 'verified_purchase_vector', 'helpful_votes_vector', 'wordCount_vector'])

aws_encoder = encoder.fit(aws_indexer).transform(aws_indexer)

aws_encoder = aws_encoder.drop('product_category_index', 'verified_purchase_index', 'helpful_votes', 'wordCount')

aws_encoder.show()

# declare tokenizer with name of output column
tokenizer = RegexTokenizer(inputCol = 'clean_review_body', outputCol="textTokens", pattern = "\\w+", gaps = False)

# transform review_body column and apply tokenizer
aws_token = tokenizer.transform(aws_encoder)

aws_token = aws_token.drop('clean_review_body')

aws_token.show()

# declare stop words remover
stop_words = StopWordsRemover(inputCol = "textTokens", outputCol = "stop_words_removed")

# transform textToken column and apply stop words remover
aws_nostop = stop_words.transform(aws_token)

aws_nostop = aws_nostop.drop('star_rating')

aws_nostop.show()

# Stem text
stemmer = SnowballStemmer(language='english')
stemmer_udf = udf(lambda tokens: [stemmer.stem(token) for token in tokens], ArrayType(StringType()))
aws_stemmed = aws_nostop.withColumn("words_stemmed", stemmer_udf("stop_words_removed"))

aws_stemmed = aws_stemmed.drop('textTokens')

aws_stemmed.show()

# perform hashingTF to prepare for IDF
hashingTF = HashingTF(inputCol="words_stemmed", outputCol="rawFeatures")
aws_tf = hashingTF.transform(aws_stemmed)

# perform TF-IDF to vectorize the data
idf = IDF(inputCol="rawFeatures", outputCol="features", minDocFreq = 1)
idfModel = idf.fit(aws_tf)
aws_tfidf = idfModel.transform(aws_tf)

aws_tfidf.select(['words_stemmed', 'rawFeatures', 'features']).show()

aws_tfidf = aws_tfidf.drop('stop_words_removed', 'rawFeatures')

aws_tfidf.show()

# write out the file as parquet to my s3 bucket
output_file_path = "s3://my-data-bucket-sd/transformed_amazon_reviews_us_ver4.parquet"
aws_tfidf.write.parquet(output_file_path)

