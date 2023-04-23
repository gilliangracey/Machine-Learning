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



def setup(path_review_train, path_product_train):
    df_train = pd.read_json(path_review_train)[["asin","unixReviewTime","verified", "vote",
                                             "reviewText","summary"]]
    #df_products = pd.read_json(path_product_train)
    #df_train = pd.merge(df_reviews, df_products, on = "asin") #asin = amazon id of product being reviewed

    #del df_products
    #del df_reviews
    
    df_train = df_train[df_train['summary'].notna() & df_train['reviewText'].notna()]
    
    return df_train



def calcWeights(sentiment, upvotes, reviewTime, verifiedReview): 

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


def scalebyVerifiedReview(verifiedReview, sentiment):
    sentimentScaleVerified = 1 if sentiment > 0 else - 1
    sentimentScaleUnverified = -.5 if sentiment > 0 else .5
    if (verifiedReview):
        sentiment += sentimentScaleVerified
    else:
        sentiment += sentimentScaleUnverified
    return float(sentiment)


def scalebyUpvotes(upvotes, sentiment):
    
    if sentiment > 0:
        sentiment += math.log(upvotes, 10) 
    else: 
        sentiment -= math.log(upvotes, 10)
    return float(sentiment)



def scalebyTime(reviewTime, sentiment):
    currentTime = time.time()
    timeWeight = 1 / (currentTime - reviewTime + 1)
    sentiment = (timeWeight + 1) * sentiment
    return float(sentiment)



def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    filtered_tokens = [token for token in tokens if token not in stopwords.words('english')]
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in filtered_tokens]
    processed_text = ' '.join(lemmatized_tokens)
    return processed_text


def get_sentiment(text):
    scores = analyzer.polarity_scores(text)
    return scores['compound']



def sentiment_analysis(text):
    #pre process
    processed_text = preprocess_text(text)

    #get sentiment
    sentiment = get_sentiment(processed_text)
    
    #return scores
    return float(sentiment)



def run_nlp(arg):
    star_scores = {"Five Stars": 0.75, "Four Stars": 0.5, "Three Stars": 0.25, "Two Stars": -0.5, "One Star": -0.75}
    asin, lst = arg

    summary_weight_sum = 0
    review_weight_sum = 0
    num_reviews = len(lst)

    for i in lst:
        #get data from rows
        review = df_train.loc[i]
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

    
    return pd.DataFrame.from_dict({"Asin": [asin], "Avg Summary Sentiment Score":[summary_weight_sum / num_reviews], "Avg Review Sentiment Score":[review_weight_sum / num_reviews]})
    


#setup 
nltk.download("all", quiet=True)
analyzer = SentimentIntensityAnalyzer()

path_review_train = "devided_dataset_v2/Toys_and_Games/train/review_training.json"
path_product_train = "devided_dataset_v2/Toys_and_Games/train/product_training.json"
df_train = setup(path_review_train, path_product_train)

if __name__ == '__main__':
    #setup training data

    #run NLP on products
    #calculate groups of indices for shared asin 
    grp_lst_args = list(df_train.groupby('asin').groups.items())

    pool = mp.Pool(processes = (mp.cpu_count() - 1))
    results = pool.map(run_nlp, grp_lst_args)
    pool.close()
    pool.join()

    results_df = pd.concat(results)
    results_df.to_csv("feature_vector.csv")
