import csv
import time
from typing import Dict

import twitter

import re
import nltk
from nltk.tokenize import word_tokenize
from string import punctuation
from nltk.corpus import stopwords

api = twitter.Api(consumer_key='B7bj4O7zXwJtp1Tfl3eKlgAOa',
                  consumer_secret='bUBnGpfsdULjglB02dRO73bJGZuSaexdGpyUH2OYQHb1vkFPUc',
                  access_token_key='1172288473476161538-Vj0DYXDyOY2yzl7efAM1V2Jg6n4KvL',
                  access_token_secret='B8uDdvwZM657s3sdODYszlhMXiVsFLoOkAzMJ6kiu1tew')

characterizationFile = "../Helpers/corpus.csv"
retrievedTweetsFile = "../Helpers/write_back_data.csv"

class tweetRetrival:
    # removes emojis and other unrecognizable elements not aligning with ascii
    def deEmojify(self, data):
        newData = []
        for tweet in data:
            txt = tweet.text.encode('ascii', 'ignore').decode('ascii')
            newData.append(txt)
        return newData

    # returns list of dictionaries containing text representing the tweets along with
    # a label specifying the tweet sentiment
    def retrieveTweets(self, query):
        try:
            tweetData = api.GetSearch(term=query, count=100)

            return [{"text": tweet, "label": None} for tweet in self.deEmojify(tweetData)]
        except:
            print("Error: Failed to retrieve twitter data")
            return None

#This class sets up the tweets for future processessing in relation to their sentiment. Meaning that we extract
#the only necessary information from the tweets that is relevant to the sentiment calculation.
class setUpTweets:
    def __init__(self):
        self._stopwords = set(stopwords.words('english') + list(punctuation) + ['AT_USER', 'URL'])

    def process(self, tweetsList, testSet):
        postProcessTweets = [];
        if(testSet):
            for tweet in tweetsList:
                postProcessTweets.append((self.__process(tweet.get("text")), tweet.get("label")))
        else:
            for tweet in tweetsList:
                postProcessTweets.append((self.__process(tweet[1]), tweet[2]))
        return postProcessTweets

    #removing words that that will now have an affect on the sentiment analysis calculation
    def __process(self, tweet):
        tweet = re.sub('((www\.[^\s]+)|(https?://[^\s]+))', 'URL', tweet)  # remove URL
        tweet = tweet.lower()  # convert to lower case
        tweet = re.sub(r'#([^\s]+)', r'\1', tweet)  # delete hashtag
        tweet = re.sub('@[^\s]+', 'AT_USER', tweet)  # delete username
        tweet = word_tokenize(tweet) #removes repeating characters in words
        return [word for word in tweet if word not in self._stopwords]

class trainAndClassify:

    #Training set which will be used to classify data
    def createTrainingSet(self, retrievalFile, trainingSetFile):
        corpusData = []
        trainingSet = []

        with open(retrievalFile, 'r') as file:
            line = csv.reader(file, delimiter=',', quotechar="\"")
            for section in line:
                corpusData.append({"id": section[2], "label": section[1], "topic": section[0]})

        #retrievedTweet: Dict[str, str]
        for retrievedTweet in corpusData:
            try:
                tweetStatus = api.GetStatus(retrievedTweet.get("id"))
                retrievedTweet["text"] = tweetStatus.text
                trainingSet.append(retrievedTweet)
                time.sleep(900 / 180);  # twitter api timeout limit for set amt of tweets retrieved
            except:
                continue
        with open(trainingSetFile, 'w') as file:
            linew = csv.writer(file, delimiter=',', quotechar="\"")
            for tweet in trainingSet:
                try:
                    linew.writerow([tweet.get("id"), tweet.get("text"), tweet.get("label"), tweet.get("topic")])
                except Exception as e:
                    print(e)
        return trainingSet

    def getTrainingSet(self, retrivalFile):
        with open(retrivalFile, 'r') as f:
            reader = csv.reader(f)
            trainingSet = list(reader)
        return trainingSet

    # extracts every distinct word from training set and then returns a key-value pairing, where the key represents the
    # frequency of that word
    def buildVocab(self, postProcessTrainingSet):
        vocabList = []
        for (words, sentiment) in postProcessTrainingSet:
            vocabList.extend(words)

        wordsList = nltk.FreqDist(vocabList)
        word_frequency_pairing = wordsList.keys()

        return word_frequency_pairing

    def tweetMatching(self, tweet):
        tweet_partition = set(tweet)
        features = {}
        for word in word_frequency_pairing:
            features['contains(%s)' % word] = (word in tweet_partition)
        return features

if __name__ == "__main__":
    api.VerifyCredentials() #API access verification

    tweet_retrival = tweetRetrival()
    train_and_classify = trainAndClassify()

    retrievedTweets = tweet_retrival.retrieveTweets(input("Enter a company name: "))
    trainingData = train_and_classify.getTrainingSet(retrievedTweetsFile)
    #trainingData = train_and_classify.createTrainingSet(characterizationFile,retrievedTweetsFile)      #creates traning set(only need to run once)

    tweetProcessor = setUpTweets()
    trainingSet = tweetProcessor.process(trainingData, False)
    testSet = tweetProcessor.process(retrievedTweets, True)

    word_frequency_pairing = train_and_classify.buildVocab(trainingSet)
    trainingFeatures = nltk.classify.apply_features(train_and_classify.tweetMatching, trainingSet)

    NBayesClassifier = nltk.NaiveBayesClassifier.train(trainingFeatures)

    NBResultLabels = [NBayesClassifier.classify(train_and_classify.tweetMatching(tweet[0])) for tweet in testSet]

