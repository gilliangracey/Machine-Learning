#!/usr/bin/env python
# coding: utf-8
import math, time, os, nltk
import multiprocessing as mp
from ctypes import Structure, c_double, c_wchar_p, c_int
from multiprocessing.sharedctypes import Value, Array
import matplotlib as plt
import pandas as pd
import numpy as np
from sklearn import model_selection
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor 
from sklearn.metrics import mean_absolute_error
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer


#setup_nlp()
#   --> takes path to reviews json and returns dataframe with desired columns
def setup_nlp(path_reviews):
    df_reviews = pd.read_json(path_reviews)[["asin","unixReviewTime","verified", "vote",
                                             "reviewText","summary"]]

    df_reviews = df_reviews[df_reviews['summary'].notna() & df_reviews['reviewText'].notna()]
    
    return df_reviews

#calcWeights()
#   --> takes sentiment analysis score along with review's upvotes, time, verification
#       and calls helper function to scale sentiment score based on these other parameters
#       helper to run_nlp()
def calcWeights(sentiment, upvotes, reviewTime, verifiedReview): 
    #check data is not Nan for other parameters
    if reviewTime is not None:
        sentiment = scalebyTime(float(reviewTime), sentiment)
    if upvotes is not None:
        upvotes = int(upvotes.replace(",",""))
        sentiment = scalebyUpvotes(upvotes, sentiment)
    if verifiedReview is not None: 
        sentiment = scalebyVerifiedReview(verifiedReview, sentiment)
    else:
        sentiment = scalebyVerifiedReview(False, sentiment)
    return sentiment

#scalebyVerifiedReview()
#   --> weights sentiment analysis score based on verification of review
#       helper to calcWeights()
def scalebyVerifiedReview(verifiedReview, sentiment):
    sentimentScaleVerified = 1 if sentiment > 0 else - 1
    sentimentScaleUnverified = -.5 if sentiment > 0 else .5
    if (verifiedReview):
        sentiment += sentimentScaleVerified
    else:
        sentiment += sentimentScaleUnverified
    return float(sentiment)

#scalebyUpvotes()
#  --> weights sentiment analysis scored based on upvotes recieved by review 
#       helper to calcWeights()
def scalebyUpvotes(upvotes, sentiment):
    
    if sentiment > 0:
        sentiment += math.log(upvotes, 10) 
    else: 
        sentiment -= math.log(upvotes, 10)
    return float(sentiment)


#scalebyUpvotes()
#  --> weights sentiment analysis scored based on how recent the review is
#       helper to calcWeights()
def scalebyTime(reviewTime, sentiment):
    currentTime = time.time()
    timeWeight = 1 / (currentTime - reviewTime + 1)
    sentiment = (timeWeight + 1) * sentiment
    return float(sentiment)


#preprocess_text()
#  --> removes stop words and process it for use by vader sentiment analysis, 
#       helper funaction to sentiment_analysis()
def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    filtered_tokens = [token for token in tokens if token not in stopwords.words('english')]
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in filtered_tokens]
    processed_text = ' '.join(lemmatized_tokens)
    return processed_text



#sentiment_analysis()
#  --> uses vader sentiment intensity analyzer to get polarity scores,
#      returning compound score,

def sentiment_analysis(text):
    #pre process
    processed_text = preprocess_text(text)

    #get sentiment
    scores = analyzer.polarity_scores(processed_text)
    
    #return compound score casted as a float 
    return float(scores['compound'])


#run_nlp()
#  --> target of each worker thread
#      arg = asin and list of indices for reviews mapped to that asin
#      calculates weighted average sentiment analysis score of review summary and review text for a given asin (product)
def run_nlp(arg):
    star_scores = {"Five Stars": 0.75, "Four Stars": 0.5, "Three Stars": 0.25, "Two Stars": -0.5, "One Star": -0.75}
    asin, indices = arg

    summary_weight_sum = 0
    review_weight_sum = 0
    num_reviews = len(indices)

    for i in indices:
        #get data from rows
        review = df_reviews.loc[i]
        summary_text = review["summary"]
        review_text = review["reviewText"]
        time = review["unixReviewTime"]
        verified = review["verified"]
        vote = review["vote"]
        
        #get sentiment analysis score for summary
        if summary_text in star_scores:
            summary_sentiment = star_scores[summary_text]
        else:
            summary_sentiment = sentiment_analysis(summary_text)
    
        #get sentiment analysis score for review text
        review_sentiment = sentiment_analysis(review_text)
    
        #weight based on review time, votes, verification
        review_weight = calcWeights(review_sentiment, vote, time, verified)
        summary_weight = calcWeights(summary_sentiment, vote, time, verified)

        summary_weight_sum += summary_weight
        review_weight_sum += review_weight

    
    return pd.DataFrame.from_dict({"asin": [asin], "Avg Summary Sentiment Score":[summary_weight_sum / num_reviews], "Avg Review Sentiment Score":[review_weight_sum / num_reviews]})
    


#SETUP of GLOBAL VARIABLES
#needs to be here so each worker thread can access the dataframe of reviews
nltk.download("all", quiet=True)
analyzer = SentimentIntensityAnalyzer()

path_review_train = "devided_dataset_v2/Toys_and_Games/train/review_training.json"
path_product_train = "devided_dataset_v2/Toys_and_Games/train/product_training.json"

df_reviews = setup_nlp(path_review_train)

if __name__ == '__main__':

    #get indices of reviews for each asin
    grouped_asin_list = list(df_reviews.groupby('asin').groups.items())

    #use pool of workers for nlp processing
    #pool.map() will split args across worker threads evenly
    pool = mp.Pool(processes = (mp.cpu_count() - 1))
    results = pool.map(run_nlp, grouped_asin_list) 
    pool.close()
    pool.join()

    #join results
    df_feature_vector = pd.concat(results).reset_index()

    #free memory 
    del df_reviews

    

    
    
    
