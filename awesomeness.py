import pandas as pd
import numpy as np
import json
import time
import csv
import math
import nltk
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor 
from sklearn.metrics import mean_absolute_error
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

nltk.download('all')
analyzer = SentimentIntensityAnalyzer()

def calcWeights(reviewSentiment, summarySentiment, upvotes, reviewTime, verifiedReview): 
    sentiment = (reviewSentiment + summarySentiment) / 2
    sentiment = scalebyTime(reviewTime, sentiment)
    sentiment = scalebyUpvotes(upvotes, sentiment)
    sentiment = scalebyVerifiedReview(verifiedReview, sentiment)
    return sentiment

    
    

def scalebyVerifiedReview(verifiedReview, sentiment):
    sentimentScaleVerified = 1 if sentiment > 0 else - 1
    sentimentScaleUnverified = -.5 if sentiment > 0 else .5
    if (verifiedReview):
        sentiment += sentimentScaleVerified
    else:
        sentiment += sentimentScaleUnverified
    return sentiment


def scalebyUpvotes(upvotes, sentiment):
    sentiment = sentiment + math.log(upvotes, 10) if sentiment > 0 else sentiment - math.log(upvotes, 10)
    return sentiment
    
def scalebyTime(reviewTime, sentiment):
    currentTime = time.time()
    timeWeight = 1 / (currentTime - reviewTime + 1)
    sentiment = (timeWeight + 1) * sentiment
    return sentiment
#rev = open('review_training.json')
scores = open('product_training.json')

#OPEN THE CSV REVIEWS FILE AND INITIALIZE TWO DICTIONARIES: ONE THAT MAPS ASINS TO THEIR REVIEWS, AND ONE THAT MAPS ASINS TO REVIEW SUMMARIES
reviews = {}
reviewsummaries = {}
with open('/Users/gilliangracey/Desktop/ML/Final/Machine-Learning/Toys_and_Games/train/train_reviews.csv', newline='') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=',')
    counter=0
    for row in spamreader:
        asin = row[1]
        review = row[7]
        summary = row[8]
        if asin in reviews:
            oldrev = reviews[asin]
            oldsum = reviewsummaries[asin]
            oldrev.append(review)
            oldsum.append(summary)
            reviews[asin] = oldrev
            reviewsummaries[asin] = oldsum
        else:
            reviews[asin] = [review]
            reviewsummaries[asin] = [summary]

def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    filtered_tokens = [token for token in tokens if token not in stopwords.words('english')]
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in filtered_tokens]
    processed_text = ' '.join(lemmatized_tokens)
    return processed_text

def get_sentiment(text):
    scores = analyzer.polarity_scores(text)
    if scores['compound']>0.5:
        sentiment = "POSITIVE"
    else:
        sentiment = "NEGATIVE/NEUTRAL"
    return sentiment





