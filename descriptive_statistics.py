# install packages
pip3 install pandas
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
import boto3
import matplotlib.pyplot as plt
import seaborn as sns
import s3fs

from pyspark.sql.functions import col,isnan, when, count, udf
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType
from pyspark.ml.stat import Correlation
from pyspark.ml.feature import VectorAssembler

# define bucket, file name, and file path
bucket = 'my-data-bucket-sd/'

file_name = 'AWS_FileNames.csv'

file_path = 's3a://' + bucket + file_name

# read the csv file from my bucket
df = spark.read.csv(file_path, sep = ',', header = True, inferSchema = True)

# declare an empty list
file_names = []

aws_bucket = 'amazon-reviews-pds/tsv/'

# loop through each row in the dataframe to get the file name
dfCollect = df.collect()
for row in dfCollect:
	aws_file_path = 's3a://' + aws_bucket + row['FileName']
	file_names.append(aws_file_path)

aws = spark.read.csv(file_names, sep = '\t', header = True, inferSchema = True)

# look at the total number of rows for the dataframe 
aws.count()

# look at the schema and column name of the dataframe
aws.printSchema()

# select only the interested columns
aws_filtered = aws.select("product_title", "product_category", "star_rating", "helpful_votes", "total_votes", \
	"vine", "verified_purchase", "review_headline", "review_body", "review_date")

# look at the summary statistics for the numerical columns
aws_filtered.select("star_rating", "helpful_votes", "total_votes").summary().show()

# look at the summary statistics of the review columns
aws_filtered.select("review_headline", "review_body").summary("count", "min", "25%", "50%", "75%", "max").show()

# define a function to strip out any non-ascii characters
def ascii_only(mystring):
	if mystring:
		return mystring.encode('ascii', 'ignore').decode('ascii')
	else:
		return None

# turn the function into a user-defined function (UDF)
ascii_udf = F.udf(ascii_only)

# apply the function to review headine and review body
aws_filtered = aws_filtered.withColumn("clean_review_headline", ascii_udf("review_headline"))
aws_filtered = aws_filtered.withColumn("clean_review_body", ascii_udf("review_body"))

# check the summary statistics for the clean review headline and review body
aws_filtered.select("clean_review_headline", "clean_review_body").summary("count", "min", "25%", "50%", "75%", "max").show()

# look at null values in each column
aws_filtered.select([count(when(isnan(c) | col(c).isNull(), c)).alias(c) for c in ["product_title", "product_category", \
	"star_rating", "helpful_votes", "total_votes", "vine", "verified_purchase", "clean_review_body", "clean_review_headline"]]).show()

# look at the null value of review date
aws_filtered.select([count(when(col(c).isNull(), c)).alias(c) for c in ["review_date"]] ).show()

# drop null values in star rating, review body, and review date columns
aws_filtered = aws_filtered.na.drop(subset = ["star_rating", "review_body", "review_date"])

# look at null values in each column
aws_filtered.select([count(when(isnan(c) | col(c).isNull(), c)).alias(c) for c in ["product_title", "product_category", \
	"star_rating", "helpful_votes", "total_votes", "vine", "verified_purchase", "clean_review_body", "clean_review_headline"]]).show()

# look at the null value of review date after dropping the null values
aws_filtered.select([count(when(col(c).isNull(), c)).alias(c) for c in ["review_date"]] ).show()

# look at the earliest and latest date in review date column
col_earlist_date = F.min('review_date').alias('earliest')
col_latest_date = F.max('review_date').alias('latest')
df_result = aws_filtered.select(col_earlist_date, col_latest_date)
df_result.show()

# create new columns based on review date
aws_filtered = aws_filtered.withColumn("review_year_month", F.date_format(col("review_date"), "yyyy-MM"))
aws_filtered = aws_filtered.withColumn("review_year", F.year(col("review_date")))

# save the necessary data to pandas data frame
review_df = aws_filtered.where(col("review_year") >= 2005).groupby("review_year_month").count().sort("review_year_month").toPandas()

# graph using matplotlib
plt.figure(figsize=(15,8))
plt.bar(review_df['review_year_month'], review_df['count'])
plt.xlabel("Year-Month")
plt.ylabel("Number of Reviews")
plt.title("Number of Reviews by Year and Month")
plt.xticks(rotation = 90, ha = "right")
plt.tight_layout()

# create a buffer to hold the figure
img_data = io.BytesIO()
plt.savefig(img_data, format = "png", bbox_inches='tight')
img_data.seek(0)

# connect to the s3fs file system
s3 = s3fs.S3FileSystem(anon = False)
with s3.open('s3://my-data-bucket-sd/review_date_barplot_ver3.png', 'wb') as f:
	f.write(img_data.getbuffer())

# show frequency of the star_rating column
star_counts_df = aws_filtered.groupby("star_rating").count().sort('star_rating').toPandas()

plt.figure(figsize = (15,8))
plt.bar(star_counts_df['star_rating'], star_counts_df['count'])
plt.title("Review Count by Star Rating")
plt.savefig(img_data, format = "png", bbox_inches='tight')
img_data.seek(0)
with s3.open('s3://my-data-bucket-sd/frequency_star_rating.png', 'wb') as f:
	f.write(img_data.getbuffer())

# select only the interested columns
aws_sim = aws_filtered.select("clean_review_body", "star_rating", "verified_purchase", "product_category", "helpful_votes")

# apply lowercase to review_body column
aws_sim = aws_sim.withColumn("clean_review_body", F.lower(col("clean_review_body")))

# remove punctuations to review_body column
aws_sim = aws_sim.withColumn('clean_review_body', F.translate('clean_review_body', '!"#$%&\'()*+,-./:;<=>?@[\\]^_{|}~', ''))

# add a new column to have number of word count
aws_sim = aws_sim.withColumn("wordCount", F.size(F.split(F.col("clean_review_body"), " ")))

# cast star_rating to double type
aws_sim = aws_sim.withColumn("star_rating", aws_sim.star_rating.cast(DoubleType()))

aws_sim.show()

# convert the numeric values to vector columns
vector_column = "correlation_features"
numeric_columns = ['star_rating', 'helpful_votes', 'wordCount']
assembler = VectorAssembler(inputCols = numeric_columns, outputCol = vector_column)
vector = assembler.transform(aws_sim)

# create the correlation matrix, then get just the values and convert to a list
matrix = Correlation.corr(vector, vector_column).collect()[0][0]
correlation_matrix = matrix.toArray().tolist()

# convert the correlation to a pandas dataframe
correlation_matrix_df = pd.DataFrame(data = correlation_matrix, columns = numeric_columns, index = numeric_columns)

# plot the correlation matrix using seaborn
plt.figure(figsize = (15,8))
sns.heatmap(correlation_matrix_df, xticklabels=correlation_matrix_df.columns.values, yticklabels=correlation_matrix_df.columns.values, cmap = 'Greens', annot = True)
plt.savefig(img_data, format = "png", bbox_inches='tight')
img_data.seek(0)
with s3.open('s3://my-data-bucket-sd/correlation_matrix.png', 'wb') as f:
	f.write(img_data.getbuffer())

# write out the file to my s3 bucket
output_file_path = "s3://my-data-bucket-sd/cleaned_amazon_reviews_us_ver2.parquet"
aws_sim.write.parquet(output_file_path)
