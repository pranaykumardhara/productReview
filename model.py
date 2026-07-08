# importing require libraries
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('always')
warnings.filterwarnings('ignore')

df=pd.read_excel('demo_data.xlsx')

# perform text preprocessing
import re #regular expression(clean the text, help in stemming or lemmatizing)
import nltk #Natural language toolkit(library for NLP)
from nltk.corpus import stopwords  # importing the stopwords
from nltk.stem import WordNetLemmatizer # this is the library for lemmatization

lemmatizer=WordNetLemmatizer()
corpus=[] # empty list

# Apply lemmatization
for i in range(0,len(df)):
    review=re.sub('[^a-zA-Z]',' ',df['Feedback_review'][i]) #remove all the characters except a-z and A-Z
    review=review.lower() #convert all the messages in lower case
    review=review.split() 
    review=[lemmatizer.lemmatize(word) for word in review if not word in stopwords.words('english')]
    review=' '.join(review)
    corpus.append(review)

# Creating the TF-IDF Model:

# We will use Term frequency and inverse document frequency(TF-IDF) instead of bag of words model.
# Because it will give better accuracy to our analysis model.

from sklearn.feature_extraction.text import TfidfVectorizer
tv=TfidfVectorizer(max_features=8000,ngram_range=(1,2))  # reduce some features to overcome model overfitting

# independent features
x=tv.fit_transform(corpus).toarray()

# dependent features

y=df['Sentiment']

# perform train test splitting
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=101) # train-test ratio is 80:20

# Implement best model for the dataset : Support Vector Machine(SVM) 
from sklearn.svm import SVC
clf = SVC(kernel='linear') # apply 'linear kernel'
clf.fit(x_train,y_train) # fit the model with traing dataset
clf.predict(x_test) # predict on test data


# serialization (creating a pickle file for the model and the features)
import pickle
pickle.dump(tv,open('tv_transform.pkl','wb')) # for features
pickle.dump(clf, open('model.pkl','wb')) # for model