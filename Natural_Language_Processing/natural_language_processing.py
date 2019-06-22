#NLP model (predicting if restaurant review is positive or negative based on text)

#importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


#importing dataset (tsv file) (beside text in training data we know whether review is positive or negative indicated by 1 or 0)
#tsv files, things are seperated by tabs (not spaces, but tabs) instead of commas like csv, we want a tsv file


dataset = pd.read_csv('Restaurant_Reviews.tsv', delimiter = '\t', quoting = 3) #converting tsv format to csv

#Cleaning the texts (Getting ride of irrelevant words) Stemming words

import re #good library for cleaning text
import nltk #good library for natural language processing
nltk.download('stopwords') #downloading all the useless ireelevant words
from nltk.corpus import stopwords #importing stopwords that we downloaded
from nltk.stem.porter import PorterStemmer #class for stemming

corpus = [] 

for i in range (0, 1000):
    review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i]) #keeping all letters including caps and non-caps
    #replacing all else with space
    review = review.lower() #replacing all with lower case
    review = review.split() #splits sentence into a list of strings
    review = [word for word in review if not word in set(stopwords.words('english'))] #looping through all strings in review list and making sure they are not useless by checkin if they are not in stopwords package
    #using set because it is a lot quicker for algorithm to read through 
    #filtered out all words NOT in stopwords list
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review] #stemming all strings in list to make sure they are the root of their parent word
    
    review = ' '.join(review) #joining list of strings into one string seperated by space as specified by ' '.join
    corpus.append(review) #appending it to our list of reviews that are clean

#Data is now clean!

#Creating bag of words model (tokenize )
from sklearn.feature_extraction.text import CountVectorizer #class to form vocab table and amount of times word is repeatede
cv = CountVectorizer(max_features = 1500) #only keeping 1500 most frequent features/words
X = cv.fit_transform(corpus).toarray() #converting to array and fitting to corpus
#Bag of words model is done, now we need to include dependent variable
y = dataset.iloc[:, 1] #this is the output

#Apply classification model (Most common is Naive Bayes or Random Forest, right now we will use Naive Bayes)

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

"""# Feature Scaling #We don't need feature scaling because there are very few features above 1 or 2
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)"""


# Fitting classifier to the Training set
# Create your classifier here

# Predicting the Test set results
# Fitting classifier to the Training set
# Create your classifier here
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 50, criterion = 'entropy') #choose how many trees (Beware overfitting)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)








