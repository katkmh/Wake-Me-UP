from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import nltk
# nltk.download('punkt')
# nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import collections
import datetime
import numpy as np
import itertools
import requests
import string
import re

# MAIN FUNCTION
# returns (bool, list)
# where boolean value answering "no classes?"
# and list contains scraped dates without classes
def scraping():
  # MAIN RETURNING
  noclasses = False

  # print("... SCRAPE training set")
  tweetsdb = scrapeFile()
  all_tweets, tweet_words = getTraining(tweetsdb)     # tokenized training data
  all_labels = labelTweets(all_tweets)    # labels of training data
  scrapeMPB()

  train_set = all_tweets[0:len(all_tweets)]
  train_labels = all_labels[0:len(all_labels)]

  ##### COUNT VECTOR
  vectorizer = CountVectorizer()
  train_cv = vectorizer.fit_transform(train_set)

  ##### TRAINING
  mnb = MultinomialNB()
  # print("... Training")
  mnb.fit(train_cv, train_labels)
  # print("... Done Training")

  # SCRAPING DUMMY ACCOUNT FOR UPDATES
  # print("... Scraping @fakemaypasokba")
  no_classes = []       # list containing no classes tweet(s)
  scraped_dates = []    # list containing dates mentioned no classes
  scraped = scrapeOnline()
  if len(scraped)>0:
    # TESTING
    scraped_set, temp = getTweets(scraped)
    scraped_cv = vectorizer.transform(scraped_set)
    scraped_predict = mnb.predict(scraped_cv)

    # GETTING TWEETS LABELED AS NO CLASSES TWEETS
    for i in range(len(scraped_predict)):
      if (scraped_predict[i]==1):
        no_classes.append(scraped[i])
    if (len(no_classes)>0):
      # there exists a tweet labelled as no classes tweet
      # extract the date mentioned in tweet
      scraped_dates = getDate(no_classes)

  # CHECK IF MAY PASOK TODAY
  today = datetime.datetime.now()
  if scraped_dates[0] == (int(today.strftime("%m")),today.day):
    scraped_dates.pop(0)
    noclasses = True

  ret = (noclasses, scraped_dates)
  return ret

# function for scraping tweets made today from dummy USC account
# returns list of tuples, each as (date posted,tweet)
def scrapeOnline():
  # handle = input('Input your account name on Twitter: ')
  mo_fil = {"Ene":"Jan","Peb":"Feb","Mar":"Mar","Abr":"Apr","May":"May","Hun":"Jun","Hul":"Jul","Ago":"Aug","Set":"Sep","Okt":"Oct","Nob":"Nov","Dis":"Dec"}
  handle="fakemaypasokba"
  # ctr = int(input('Input number of tweets to scrape: '))
  ctr = 20
  res=requests.get('https://twitter.com/'+ handle)
  bs=BeautifulSoup(res.content,'lxml')
  all_tweets = bs.find_all('div',{'class':'tweet'})
  tweets = []


  if all_tweets:
    for tweet in all_tweets[:ctr]:
      # SCRAPING TWEETS
      context = tweet.find('div',{'class':'context'}).text.replace("\n"," ").strip()
      content = tweet.find('div',{'class':'content'})
      header = content.find('div',{'class':'stream-item-header'})
      user = header.find('a',{'class':'account-group js-account-group js-action-profile js-user-profile-link js-nav'}).text.replace("\n"," ").strip()
      time = header.find('a',{'class':'tweet-timestamp js-permalink js-nav js-tooltip'}).find('span').text.replace("\n"," ").strip()
      message = content.find('div',{'class':'js-tweet-text-container'}).text.replace("\n"," ").strip()
      footer = content.find('div',{'class':'stream-item-footer'})
      stat = footer.find('div',{'class':'ProfileTweet-actionCountList u-hiddenVisually'}).text.replace("\n"," ").strip()

      # check for new tweets made TODAY
      times = time.split()
      today = datetime.datetime.now()
      if times[0] in mo_fil:
        if mo_fil[times[0]]!=today.month:
          if times[1]!=today.day:
            break

      # DEBUG: PRINTING VALUES
      # if context:
      #   print("1: "+context)
      # print("2: "+time)
      # print("3: "+message)
      # print("4: "+stat)
      # print()

      # ADDING TO DB OF TWEETS (tweets)
      tweets.append([time,message])
  else:
      print("List is empty/account name not found.")

  return tweets

# function for scraping dates when disabled
# param: tweets is a list of no classes tweets
def getDate(tweets):
  month = {"Enero":"January","Pebrero":"February","Marso":"March","Abril":"April","Mayo":"May","Hunyo":"June","Hulyo":"July","Agosto":"August","Setyembre":"September","Oktubre":"October","Nobyembre":"November","Disyembre":"December"}
  mo = {"Jan":1,"Feb":2,"Mar":3,"Apr":4,"May":5,"Jun":6,"Jul":7,"Aug":8,"Sep":9,"Oct":10,"Nov":11,"Dec":12}
  dates = []
  for tweet in tweets:
    string = tweet[1]
    string = string.replace('.',' ')
    string = string.replace('-',' - ')
    string = string.split()
    for i in range(len(string)):
      if string[i] in month:
        # tagalog month
        m = mo[month[string[i]][:3]]
        dates.append((m,int(string[i+1])))
        i+=2
        # check for range
        if string[i] == "-" or string[i] == "to":
          start = int(string[i-1])
          if string[i+1] in month or string[i+1] in month.values() or string[i+1] in mo:
            end = int(string[i+2])
            i+=2
          else:
            end = int(string[i+1])
            i+=2
          ctr = start+1
          while ctr <= end:
            dates.append((m,ctr))
            ctr+=1
      elif string[i] in month.values() or string[i] in mo:
        m = mo[string[i][:3]]
        dates.append((m,int(string[i+1])))
        i+=2
        # check for range
        if string[i] == "-" or string[i] == "to":
          start = int(string[i-1])
          if string[i+1] in month or string[i+1] in month.values() or string[i+1] in mo:
            end = int(string[i+2])
            i+=2
          else:
            end = int(string[i+1])
            i+=2
          ctr = start+1
          while ctr <= end:
            dates.append((m,ctr))
            ctr+=1
  dates.sort()
  dates = list(dict.fromkeys(dates))
  return dates

# function for scraping training set
def scrapeFile():
  f = open("USCUPDiliman_800_loaded.html", "r", encoding="utf8")
  contents = f.read()

  bs = BeautifulSoup(contents, 'lxml')

  all_tweets = bs.find_all('div',{'class':'tweet'})
  tweets = []

  if all_tweets:
    for tweet in all_tweets:
      # SCRAPING MESSAGES ONLY
      hentry = tweet.find('div',{'class':'hentry'})
      msg = hentry.find('span',{'class':'entry-content'}).text.replace("\n"," ").strip()
      # print(msg)
      tweets.append(msg)
  else:
    print("None read")

  return tweets

# function for scraping maypasokba.com
# returns -1 if not useful scrape, 0 if no classes, 1 otherwise
def scrapeMPB():
  res=requests.get('https://maypasokba.com/')
  bs=BeautifulSoup(res.content,'lxml')

  update = bs.find('h1').text.replace("\n"," ").strip()
  
  update = update.split()

  if len(update)==1:
    if update=="wala": return 0
    else: return 1
  else:
    return -1

# function for generating training dataset
# removal of special characters, lowercase, and stemming
def getTraining(tweetsdb):
  all_tweets_tokenized = []
  all_tweets = []
  tempstr = ""
  stop_words = stopwords.words('english')
  for tweet in tweetsdb:
    # TOKENIZATION
    # remove punctuation marks
    # tempstr = tweet[1].translate(tweet[1].maketrans('', '', string.punctuation))
    tempstr = re.sub('-',' ',tweet)      # for ranges and combo words
    tempstr = re.sub(r'[^\w\s]','',tempstr) # for all special characters
    # remove links
    tempstr = re.sub("http+\w*","",tempstr)
    tempstr = re.sub("pictwitter+\w*","",tempstr)
    tempstr = re.sub("pbstwimg+\w*","",tempstr)
    tempstr = re.sub("www+\w*","",tempstr)
    tempstr = re.sub("tinyurl+\w*","",tempstr)
    # convert strings to lower
    tempstr = tempstr.lower()
    tempstr = tempstr.split()
    # stemming words
    temptweet = []
    for word in tempstr:
      temptweet.append(PorterStemmer().stem(word))
    # remove stop words
    temptweet = [word for word in temptweet if word not in stop_words]
    # remove duplicates
    temptweet = list(dict.fromkeys(temptweet))

    all_tweets_tokenized.append(temptweet)
    all_tweets.append(" ".join(temptweet))

  return all_tweets, all_tweets_tokenized

# function for generating list of data sans date posted and returns it
# removal of special characters, lowercase, and stemming
def getTweets(tweetsdb):
  all_tweets_tokenized = []
  all_tweets = []
  tempstr = ""
  stop_words = stopwords.words('english')
  for tweet in tweetsdb:
    # TOKENIZATION
    # remove punctuation marks
    # tempstr = tweet[1].translate(tweet[1].maketrans('', '', string.punctuation))
    tempstr = re.sub('-',' ',tweet[1])      # for ranges and combo words
    tempstr = re.sub(r'[^\w\s]','',tempstr) # for all special characters
    # remove links
    tempstr = re.sub("http+\w*","",tempstr)
    tempstr = re.sub("pictwitter+\w*","",tempstr)
    # convert strings to lower
    tempstr = tempstr.lower()
    tempstr = tempstr.split()
    # stemming words
    temptweet = []
    for word in tempstr:
      temptweet.append(PorterStemmer().stem(word))
    # remove stop words
    temptweet = [word for word in temptweet if word not in stop_words]
    # remove duplicates
    temptweet = list(dict.fromkeys(temptweet))

    all_tweets_tokenized.append(temptweet)
    all_tweets.append(" ".join(temptweet))

  return all_tweets, all_tweets_tokenized

# function only determines tweet classification
# returns training labels: no classes tweet (1) or not (0)
def labelTweets(tweets):
  # ncDictL<n> = dictionary of nth level straining
  ncDict0 = ["holiday", "walangpasok"]
  ncDictL1 = ["no", "suspend", "cancel", "wala", "walang"]
  ncDictL2 = ["class", "classes", "pasok", "klase"]
  labels = []

  for tweet in tweets:
    # print(tweet)
    # print()
    if any(x in tweet for x in ncDict0):
      labels.append(1)
    else:
      if any(x in tweet for x in ncDictL1):
        if any(y in tweet for y in ncDictL2):
          labels.append(1)
        else:
          labels.append(0)
      else:
        labels.append(0)

  return labels

  #   # STRAIN 1st level
  #   if any(s in tweet[1] for s in ncDictL1):
  #     # STRAIN 2nd level
  #     if any(t in tweet[1] for t in ncDictL2):
  #       labels.append(1)
  #   else:
  #     labels.append(0)

  # return labels

print(scraping())
  