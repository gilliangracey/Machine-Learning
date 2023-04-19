import json
import time
import math
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

def varmean():
    for i in scores['awesomeness']:
        print(i)
    return


varmean()


def cor_coeff(variable, awesomeness):
    return 0
