# -*- coding: utf-8 -*-
"""
Created on Sun Sep  1 23:42:02 2019

@author: Taha
"""

"""  
**************    Business Case  *****************************

We are working for a client who wants to open a restaurant in Bangalore and he wants us to use Machine
Learning to help him setup his business at the right location, serving the right menu at the right price
point.As a data scientist you are tasked with the responsibility to study the data and help answer some
of the questions written below.
1.What is the best location in Bangalore to open a restaurant? Why?
2.Help him choose a cuisine for the restaurant? Why?
3.Create a Machine Learning model for them that would look at the reviews posted by their customers 
  and tell them their ranking in the city.

**************************************************************
"""

## Import Required Libraries

import pandas as pd
import numpy as np
import ast
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')

"""
***************************************************************
            Importing the Dataset and Doing EDA.
*************************************************************** 
"""


zomato_data = pd.read_csv('zomato.csv')                                          ## Import the Dataset

zomato_data.columns                                                              ## Get Column Names
zomato_data.head()                                                               ## Analyze first few rows
zomato_data.tail()                                                               ## Analyze last few rows
zomato_data.shape                                                                ## Get dataset shape
zomato_data.info()                                                               ## We have null values in *rate, phone, location, rest_type
                                                                                 ## dish_liked, cuisines, approx_cost(for two people)
zomato_data.describe()                                                           ## Only votes column is Numeric

zomato = zomato_data.copy()                                                      ## Create a Copy of Data

zomato.drop(['url', 'address','phone','name'],axis=1, inplace =True)             ## remove unwanted columns 

zomato = zomato.rename(columns = {"approx_cost(for two people)" : "avg_cost", "listed_in(type)" : "meal_type", 
                   "listed_in(city)" : "city"})                                  ## renaming columns to a better name

zomato['city'].value_counts()                                                    ## 30 distinct city values
zomato['location'].value_counts()                                                ## 93 distinct location values
zomato['city'].isnull().sum()                                                    ## City has No Null Values
zomato['location'].isnull().sum()                                                ## Location has 21 NULL values.

""" Since city attribute is better than location attribute hence we delete location column """

del zomato['location']

zomato.duplicated().sum()                                                        ## Find no of duplicate values in our dataset. 422 rwos are duplicate

zomato = zomato.drop_duplicates(subset=None, keep='first',inplace=False)         ## remove 422 duplicated rows.

""" Analyzing rate column """

zomato.rate.unique()                                                             ## We have values like '-' and 'NEW' which should be removed.
zomato.rate.replace(('NEW','-'),np.nan,inplace =True)
zomato.rate = zomato.rate.astype('str')                                          ##  make it as string
zomato.rate = zomato.rate.apply(lambda x: x.replace('/5','').strip())            ## remove the "/5" 
zomato.rate = zomato.rate.astype('float')                                        ## convert column type to float

""" Analyze review list column """

type(zomato.reviews_list[0])                                                     ## column is of type string instead of being a tuple.

zomato.reviews_list = zomato.reviews_list.apply(lambda x:ast.literal_eval(x))    ## convert it into Tuple
zomato.reviews_list[0][0]                                                        ## see the first tuple value

""" Change online order and book table column to Boolean instead of Yes / NO """

zomato.online_order.replace(('Yes','No'),(True,False),inplace =True)
zomato.book_table.replace(('Yes','No'),(True,False),inplace =True)

""" Find how many null values in diff columns and delete the rows if possible """

zomato.isnull().sum()

zomato.dropna(subset=['rate', 'avg_cost','cuisines','rest_type'],inplace=True)  ## delete rows having null values in column - rate, avg_cost,cuisines,rest_type.
                                                                                ## cant remove null records from dish_like column as there are too many records.

""" Analyze Average cost column """
zomato.avg_cost.unique()

zomato.avg_cost = zomato.avg_cost.apply(lambda x: int(x.replace(',','')))       ## remove ',' from value

## View top few rows of the final dataset.

zomato.head()

"""
***************************************************************
         Visualize the Data Insights
***************************************************************
"""
""" Analyze Locations  """ 

zomato['city'].nunique()                                                        ## 30 distincts location are present.

zomato['city'].value_counts()[:10]                                              ## Analyze top 10 locations having highest number of restaurants.

plt.figure(figsize=(12,8)) 
zomato['city'].value_counts()[:10].plot(kind = 'bar')
plt.title('Location wise Count', weight = 'bold')


""" Analyze Restaurant Types  """ 

zomato['rest_type'].nunique()                                                   ## 87 distincts Restaurants Type are present.
zomato['rest_type'].value_counts()[:10]                                         ## Analyze top 10 Restaurants Type

plt.figure(figsize=(12,8)) 
zomato['rest_type'].value_counts()[:10].plot(kind = 'bar')
plt.title('Restaurant Type wise Count', weight = 'bold')

""" Analyze Cuisines Types  """ 

zomato['cuisines'].nunique()                                                    ## 2367 which is wrong.Many restaurants are having multiple Cuisines.
                                                                                ## hence Breaking each of them and then counting.
zomato.cuisines = zomato.cuisines.apply(lambda x:x.lower().strip())
cuisines_data = zomato

cuisines_count= []

for i in cuisines_data.cuisines:
    for j in i.split(','):
        j = j.strip()
        cuisines_count.append(j)
                                                                                
plt.figure(figsize=(15,8))                                                      ## Plotting top 10 Cuisines
pd.Series(cuisines_count).value_counts()[:10].plot(kind='bar',color= 'cyan')
plt.title('Top 10 cuisines in Bangalore',weight='bold')
plt.xlabel('cuisine')
plt.ylabel('Count')                                                             ## North Indian & Chines are the 2 top Cuisines.

""" Analyze Average Cost for 2 People  """ 

zomato['avg_cost'].nunique()                                                    ## 63 Distinct avg costs                            
zomato['avg_cost'].value_counts()[:10]                                          ## Most of the restaurants have Avg cost = 300/400 RS

zomato.avg_cost.hist(color='cyan',bins = 10,range = (0,3000))                                                
plt.axvline(x= zomato.avg_cost.mean(),ls='--',color='Red')
plt.title('Avg Cost for 2',weight='bold')
plt.xlabel('Cost for 2')
plt.ylabel('No of Restaurants')
print(zomato.avg_cost.mean())                                                   ## Avg cost for 2 people among all restaurants is around 600

""" Analyze Rating of restaurant.  """ 

zomato['rate'].nunique()                                                        ## 31 unique ratings

zomato.rate.hist(color='cyan')                                                  ## Plot ratings and find AVG rating of restaurants
plt.axvline(x= zomato.rate.mean(),ls='--',color='Red')
plt.title('Avg Rating of Restaurants',weight='bold')
plt.xlabel('Rating')
plt.ylabel('No of Restaurants')
print(zomato.rate.mean())                                                       ## Avergae rating of restaurant is 3.7


""" Q1 -- What is the best location in Bangalore to open a restaurant? Why? """

zomato['city'].value_counts()[:5]                                               ## Top 5 locations having maximum number of restaurants
                                                                          
## They are - BTM , Koramangala 7th Block , Koramangala 4th Block, Koramangala 5th Block, Koramangala 6th Block

zomato_location = zomato.loc[(zomato.city == 'BTM') | (zomato.city == 'Koramangala 7th Block') | (zomato.city == 'Koramangala 4th Block') |
           (zomato.city == 'Koramangala 5th Block') | (zomato.city == 'Koramangala 6th Block')]
                                                                                ## Filter data for these 5 localities only.            
zomato_location.groupby('city')['avg_cost'].mean()                              ## Average cost for 2 people in these 5 localities.

zomato_location.groupby('city')['rate'].median()                                ## Median Rating for restaurants in these locations

zomato_location.groupby('city')['votes'].mean()                                 ## Average Votes given to restaurants in these area

""" Koramangala 5th Block' should be the area to open the restaurants. Because it has the fourth highest number of restaurants
in the area. Also the restaurants in this area has the highest Mean of Avg_cost for 2 people, and has an Average rating of 3.8
which is slightly higher than average rating of the entire population. Not only that it has the highest number of votes 
received from the people amongst the top five areas considered """

""" Q2 -- Help him choose a cuisine for the restaurant? Why?  """

cuisines_count                                                                  ## Already created a cuisines dataset    

pd.Series(cuisines_count).value_counts()[:10]                                   ## Count of top 10 Cuisines in bangalore

cuisines_data_top5 = zomato_location.copy()                                     ## Find top cuisines in the top 5 localities.
                                                                                ## North Indian and Chines are the top most Cuisines.    
cuisines_count_top5= []

for i in cuisines_data_top5.cuisines:
    for j in i.split(','):
        j = j.strip()
        cuisines_count_top5.append(j)


plt.figure(figsize=(15,8))                                                      ## Plotting top 10 Cuisines
pd.Series(cuisines_count_top5).value_counts()[:10].plot(kind='bar',color= 'cyan')
plt.title('Top 10 cuisines in top 5  locality',weight='bold')
plt.xlabel('cuisine')
plt.ylabel('Count')                                                             ## North Indian and chinese is the most common Cuisine in the top 5 localities also.       

""" The new restaurant owner should choose a combination of North Indian and Chinese Cuisines as they are the most prevalant
cuisines in entire bangalore and even in the top 5 localities of bangalore considered. """

""" 3Q -- Create a Machine Learning model for them that would look at the reviews posted by their customers 
  and tell them their ranking in the city. """

model_dataset = zomato.copy()

model_dataset = model_dataset.reset_index(drop=True)                            ## Reset the Index.

model_dataset['rate'].value_counts()                                            ## Least rating is 1.8, Max is 4.9    

""" Create  a Function to assign Rating level to each restaurant and categorise the restaurant as following
1 - Poor       (rating value between 0-2)
2 - OK         (rating value between 2-3)
3 - Good       (rating value between 3-4)
4 - Very Good  (rating value between 4-4.5)
5 - Excellent  (rating value between 4.5-5)  """

def rate_cond(df):

    if (df['rate'] >= 0) and (df['rate'] <= 2):
        return 1
    elif (df['rate'] > 2) and (df['rate'] <= 3):
        return 2
    elif (df['rate'] > 3) and (df['rate'] <= 4):
        return 3
    elif (df['rate'] > 4) and (df['rate'] <= 4.5):
        return 4
    elif (df['rate'] > 4.5) and (df['rate'] <= 5):
        return 5
    elif (df['rate'] < 0) and (df['rate'] > 5):
        return np.nan

model_dataset['rating'] = model_dataset.apply(rate_cond, axis = 1)

model_dataset = model_dataset[['rating','reviews_list']]                        ## Keep only required column.
model_dataset['rating'].value_counts()                                          ## See count of different ratings  present    
model_dataset.reviews_list[0][0]                                                ## See first review

model_dataset['review_final'] =  model_dataset['reviews_list'].apply(lambda x: ', '.join(map(str, x)))
                                                                                ## convert review column to string

"""  Cleaning the texts Before building Bag Of Words Model"""
## Import Required Modules

import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

corpus = []
for i in range(0, 41225):                                                       ## Take a subset (#20000), due to lesser memory config of PC
    review = re.sub('[^a-zA-Z]', ' ', model_dataset['review_final'][i])         ## remove everythign except characters
    review =  re.sub(r"\b[a-zA-Z]\b", '', review)                               ## remove single letter characters
    #review = re.sub(r'(?:^| )\w(?:$| )', ' ', review).strip()                  ## remove single letter characters
    review = review.lower()                                                     ## convert string to lower
    review = review.split()
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))] ## Convert the root word
    review = ' '.join(review)                                                   ## Join the words back
    corpus.append(review)

# Creating the Bag of Words model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 10000)
X = cv.fit_transform(corpus).toarray()
y = model_dataset.iloc[:, 0].values
#y = y[0:17009]


from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0) # Splitting the dataset into the Training set and Test set

from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline    import Pipeline
from sklearn.feature_extraction.text import TfidfTransformer

nb = Pipeline([('tfidf', TfidfTransformer()), ('clf', MultinomialNB()),])

nb.fit(X_train, y_train)                                                        ## Train the Model.

from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score

y_pred_nb = nb.predict(X_test)                                                  ## Do Prediction on test dataset 

print('accuracy %s' % accuracy_score(y_pred_nb, y_test))                        ## Acccuracy = 
print(classification_report(y_test, y_pred_nb))                                 ## Analyze other Model evaluation parametrs

""" Predicting Using  Linear SVM """

from sklearn.linear_model import SGDClassifier

sgd = Pipeline([('tfidf', TfidfTransformer()),('clf', SGDClassifier(loss='hinge', penalty='l2',alpha=1e-3, random_state=42, max_iter=5, tol=None)),])
sgd.fit(X_train, y_train)                                                       ## Train the Model.              


y_pred_svm = sgd.predict(X_test)                                                ## Do Prediction on test dataset

print('accuracy %s' % accuracy_score(y_pred_svm, y_test))                       ## Accuracy = 
print(classification_report(y_test, y_pred_svm))                                ## Analyze other Model evaluation parametrs


""" Predicting Using  Logistic Regression """

from sklearn.linear_model import LogisticRegression

logreg = Pipeline([('tfidf', TfidfTransformer()), ('clf', LogisticRegression(n_jobs=1, C=1e5)),])
logreg.fit(X_train, y_train)                                                    ## Train the Model.

y_pred_lr = logreg.predict(X_test)                                              ## Do Prediction on test dataset

print('accuracy %s' % accuracy_score(y_pred_lr, y_test))                        ## Accuracy = 
print(classification_report(y_test, y_pred_lr))                                 ## Analyze other Model evaluation parametrs


""" Save All the Models to disk to be able to Reuse it """

import pickle

filename_LR = 'LogReg_model.sav'                                                ## Save Log Reg Model File
pickle.dump(logreg, open(filename_LR, 'wb'))

filename_SVM = 'SVM_model.sav'                                                  ## Save SVM Model File
pickle.dump(sgd, open(filename_SVM, 'wb'))

filename_NB = 'NB_model.sav'                                                    ## Save Naive Bayes Model File        
pickle.dump(nb, open(filename_NB, 'wb'))
 
## Code to reload the Logistic regression Model from disk and run it.

loaded_model = pickle.load(open(filename_LR, 'rb'))
result = loaded_model.score(X_test, y_test)
print(result)
