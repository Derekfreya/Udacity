import sys
import pandas as pd
import numpy as np
import sqlite3
import nltk
nltk.download(['punkt', 'stopwords', 'wordnet'])
import re
import pickle
from sqlalchemy import create_engine
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from gensim.test.utils import common_texts
from gensim.sklearn_api import W2VTransformer
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report


def load_data(database_filepath):
    
    # load data from database
    con = sqlite3.connect(database_filepath)
    df = pd.read_sql("SELECT * FROM DisasterResponseRawData", con)
    
    # Set features as X and label as y
    X = df.message.values
    y = df.iloc[:, 4:].values
    
    # Use the label column names as category name
    category_names = [col for col in df.iloc[:, 4:].columns]
    
    return X, y, category_names


def tokenize(text):
    
    # tokenize each message as single words
    tokenized = word_tokenize(text)
    
    # remmove the stopwords
    stopwords_removed = [w for w in tokenized if w not in stopwords.words('english')]
    
    # lemmatize the words
    lemmed = [WordNetLemmatizer().lemmatize(w) for w in stopwords_removed]
    
    # stemming of words
    stemmed = [PorterStemmer().stem(w) for w in lemmed]
    
    return stemmed

def build_model():
    
    # build pipeline using bag of words, tfidf, and KNeigbours algorithm for the model 
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer = tokenize, 
                                ngram_range = (1, 2))),
        ('tfidf', TfidfTransformer()),
        ('KNC', KNeighborsClassifier(weights = 'distance', 
                                     algorithm = 'auto'))
    ])
    
    # grid search parameter to fine tune model
    param_grid = {
        'KNC__n_neighbors': [6, 7], 
    }

    cv = GridSearchCV(pipeline, 
                      param_grid = param_grid, 
                      scoring = 'f1_samples')
    
    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    
    # predict multlabels (categories) for each message
    Y_pred = model.predict(X_test)
        
    # setting labels to each category
    labels = [n for n in range(1, 37)]

    # run classification report on each category
    for count, column in enumerate(Y_pred.T):   
        print(classification_report(Y_test.T[count], 
                                    column, 
                                    labels = [1, 0], 
                                    target_names = [category_names[count], 'Not '+ category_names[count]]
                                   ))


def save_model(model, model_filepath):
    
    # save model to model_filepath
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()