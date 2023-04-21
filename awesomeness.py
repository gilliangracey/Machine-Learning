import pandas as pd
import numpy as np
import json
import time
import csv
import math
#import nltk
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor 
from sklearn.metrics import mean_absolute_error



#nltk.download()

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
#scores = open('product_training.json')
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
    #print(reviewsummaries['0F6D86162E84EDFB35E140A9F3BA2A1C'])