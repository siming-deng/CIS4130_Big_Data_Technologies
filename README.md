# Star Rating Prediction on Amazon Customer Reviews Dataset using NLP and ML

## Project Description

The goal of this project is to use Natural Language Processing and Machine Learning algorithms to predict Amazon’s product ratings from the customers' review comments and other related variables. Exploratory Data Analysis will also perform before applying any models to understand the dataset better. 

## Data Source

Amazon Customer Reviews Dataset is one of the public datasets available on AWS. It provided over 130+ million customer reviews for products sold on Amazon from 1995 until 2015. The reviews generated by Amazon customers describe their experience with the products offered. This dataset is in the *amazon-reviews-pds* S3 bucket in AWS US East Region. The Amazon Customer Review Dataset is available in tsv and parquet files. We can access the dataset using the AWS Command Line Interface. The location of the tsv files is *s3://amazon-reviews-pds/tsv/* and the location of the parquet files is *s3://amazon-reviews-pds/parquet/*. Each line in the data files corresponds to an individual product review. 

The dataset has the following attributes:

* **marketplace (string)**: 2-letter country code of the marketplace where the review was written
* **customer_id (integer)**: the random identifier that can be used to aggregate reviews written by a single author
* **review_id (string)**: the unique id of the review
* **product_id (string)**: the unique product id the review pertains to
* **product_parent (integer)**: the random identifier that can be used to aggregate reviews for the same product
* **product_title (string)**: title of the product
* **product_category (string)**: broad product category that can be used to group reviews
* **star_rating (string)**: the 1-5 star rating of the review
* **helpful_votes (integer)**: number of helpful votes
* **total_votes (integer)**: number of total votes the review received
* **vine (string)**: review was written as part of the Vine program
* **verified_purchase (string)**: the review is on a verified purchase or not
* **review_headline (string)**: the title of the review
* **review_body (string)**: the review text
* **review_date (timestamp)**: the date the review was written

## File Directory

### descriptive_statistics.py
> Location of the code for descriptive statistics of the dataset (EDA).
### data_preprocessing.py
> Location of the code for data preprocessing before fitting a machine learning model.
### machine_learning_models.py
> Location of the code for machine learning models and evaluation of the performance.
### cis_4130_projects_6&7_Siming_Deng.pdf
> Location of the final report of the project.

## Summary of the Dataset and Challenges had in Cleaning and Feature Engineering

The Amazon Customers Review Dataset has a total of 150,962,278 rows of records with 15 columns. All features contain null values with review_body having the highest amount of null values (18,788) and our target variable—star_raing—has 2263 null values. Since 18,788 compared to our whole dataset is a small amount of data and no other way to obtain the data back, dropping the null values seems necessary in this case. After dropping the null values for review_body and star_rating, there are still 384 null values for review_headline and 4 null values for product_title. 

After running descriptive statistics for star_rating, helpful_votes, and total_votes, the average star_rating is 4.2 with a standard deviation of 1.3 and maximum star_rating 5; the average helpful_votes is 1.9 with a standard deviation of 19.5 and maximum helpful_votes of 47,524; the average total_votes is 2.5 with a standard deviation of 21.2 and maximum total_votes of 48,362. Based on the statistics, star_rating seems to have a left-skewed distribution, and helpful_votes and total_votes seem to have right-skewed distribution.

The review_date column has a minimum date of 1995-06-24 and a maximum date of 2015-08-31. So, this dataset contains 20 years of customer reviews for various product categories. In addition, based on the summary statistics for review_headline and review_body there seem to have emojis and other characters that are not text. Therefore, removing those extraneous emojis and characters is also necessary.

In terms of the text-mining process, there will be 4 steps:

**1. Preprocess data:** 
  - Convert text (review_body) to lowercase, remove unnecessary punctuations, etc. 
  - Tokenization will also apply, which will convert sentences to words.
  - Remove stop words.
  - Stemming: Snowball Stemmer
  
**2. Vectorize data:** 
  - Word embedding: TF-IDF
  
**3. Feature engineering:**
  - Word count for review_body
  
**4. Train classifier:**
  - Logistic Regression

## Coding and Machine Learning

The goal of this project is to predict Amazon’s product star ratings based on the customers’ review comments and other related factors. Based on the observation from exploratory data analysis, we can see that there are mostly star ratings of 5 and very few star ratings of 1-4. Since most people would consider a star rating of 4+ a good product, it is better to consider predicting the star ratings as a binary classification problem with star ratings 1-3 labeled as 0 and star ratings 4-5 labeled as 1.

After looking at the variables in the dataset and the correlation matrix, there are a few variables selected to be the predictors and star_rating being the dependent variable. The predictor variables considered are review_body, verified_purchase, product_category, and helpful_votes. Another predictor variable is the new feature generated as the word count of review_body.

For a binary classification problem, we do have a couple of machine learning algorithms to choose from. For example, logistic regression is one of the most popular and common machine learning algorithms used to solve binary classification problems. In logistic regression, a logit transformation is applied to the odds—that is, the probability of success divided by the probability of failure. This is commonly known as the log odds or the natural logarithm of odds. For the purpose of the project, a logistic regression model is applied to predict the star ratings, whether it is <=3 or >3. 

## Project Summary and Main Conclusions

Logistic Regression is used to classify whether the star rating is <=3 or >3 using predictors such as the review body text, helpful_votes, product_categories, verified_purchase, and word counts. Hyperparameter tuning is implemented by setting regParam = [0, 0.2, 0.4, 0.6, 0.8, 1] and elasticNetParam = [0, 0.5, 1]. The total number of models being tested is 18. 

Our best model has an elasticNetParam of 0 and regParam of 0.2. That means that our model is reduced to a ridge regression model. The best model after hyperparameter tuning has an AUC of 0.89 with an accuracy of 0.83, a precision of 0.83, a recall of 0.99, and an f1 score of 0.9. This means that this model has a good performance on recall—the model can classify 99% of all the >3 ratings correctly. If we want to focus on predicting >3 ratings correctly then this would be a good model for that. If we want to focus on getting higher correct prediction for all predicted positive samples then we want to have higher precision. Depending on what purpose we want to accomplish we can modify and/or improve our model.

**Some future improvements include**

1. Try different machine learning models and hyperparameter tuning (e.g. Random Forest Classifier and/or Multinomial Naive Bayes)
2. Increase the sample size, but that would also increase the computational cost
3. Try different data preprocessing methods for text data (e.g. Lemmatization)
