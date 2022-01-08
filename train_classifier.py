import joblib
import nltk
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger', 'stopwords'])

import pandas as pd
import numpy as np
from sqlalchemy import create_engine

import re
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk import pos_tag, ne_chunk

from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import GridSearchCV
from sklearn.base import BaseEstimator, TransformerMixin

from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.naive_bayes import MultinomialNB


from sklearn.metrics import classification_report

import warnings
warnings.filterwarnings('ignore')

df = pd.read_sql_table('DisasterResponse', 'sqlite:///DisasterResponse.db')
X = df['message']
y = df.drop(['id', 'message', 'original', 'genre'], axis=1)


def tokenize(text):
    '''
        This function tokenizes the text

        params: text: string
        return: the string tokenized
    '''
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        if clean_tok not in stopwords.words('english'):
            clean_tokens.append(clean_tok)

    return clean_tokens


def tokenize_with_stopwords(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def display_results(y_test, y_pred):
    '''
        This function displays the f1 score, recall, precision of the result.

        y_test: array of test data
        y_pred: array of predict data
    '''
    columns = y_test.columns
    for i in range(len(columns)):
        print(columns[i])
        print(classification_report(y_pred[:, i], y_test[columns[i]]))


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)


def rf_pipeline_tf():
    '''
       Pipeline of feature extraction and classifier
               feature extraction: tf-idf transformer
               classifier: RandomForestClassifier
    '''
    pipeline = Pipeline([

        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(estimator=RandomForestClassifier(n_estimators=100)))
    ])

    parameters = {'clf__estimator__n_estimators': [10, 100]
                  }

    cv = GridSearchCV(pipeline, param_grid=parameters, cv=2)

    return pipeline


def rf_pipeline_tfv():
    '''
           Pipeline of feature extraction and classifier
                  feature extraction: tf-idf vectorizor
                  classifier: RandomForestClassifier
    '''
    pipeline = Pipeline([


        ('tfv', TfidfVectorizer()),
        ('clf', MultiOutputClassifier(estimator=RandomForestClassifier(n_estimators=100)))
    ])

    parameters = {'clf__estimator__n_estimators': [10, 100]
                  }

    cv = GridSearchCV(pipeline, param_grid=parameters, cv=2)

    return pipeline


class to_dense(BaseEstimator, TransformerMixin):
    '''
        transform the feature matrix into dense matrix
    '''
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.todense()


def mnb_pipeline():
    '''
              Pipeline of feature extraction and classifier:
                  feature extraction: tf-idf transformer
                  classifier: Multinomial Naive Bayes
    '''
    pipeline = Pipeline([

        ('vect', CountVectorizer(tokenizer=tokenize)),

        ('tfidf', TfidfTransformer()),
        ('td', to_dense()),
        ('clf', MultiOutputClassifier(MultinomialNB()))
    ])

    #     parameters = {  'clf__estimator__n_estimators': [100]
    #                  }

    #     cv = GridSearchCV(pipeline, param_grid=parameters)

    return pipeline


def svm_pipeline():
    '''
       Pipeline of feature extraction and classifier
               feature extraction: tf-idf transformer
               classifier: SVM
    '''

    pipeline = Pipeline([

        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(svm.SVC(gamma='auto')))
    ])

    parameters = {
        'clf__estimator__C': [1, 10]
    }

    cv = GridSearchCV(pipeline, param_grid=parameters, cv=2)

    return cv

def train_and_display(num):
    '''
    This function trains the model and saves it as  a pkl file, also displays its scores.
    :param num: int 1,2,3,4 for 4 different models
    '''
    if num == 1:
        model = rf_pipeline_tf()
        model.fit(X_train, y_train)
        joblib.dump(model, 'rf_pipeline_tf.pkl')

    elif num == 2:
        model = rf_pipeline_tfv()
        model.fit(X_train, y_train)
        joblib.dump(model, 'rf_pipeline_tfv.pkl')

    elif num == 3:
        model = mnb_pipeline()
        model.fit(X_train, y_train)
        joblib.dump(model, 'mnb_pipeline.pkl')

    elif num == 4:
        model = svm_pipeline()
        model.fit(X_train, y_train)
        joblib.dump(model, 'svm_pipeline.pkl')

    y_pred = model.predict(X_test)

    #     display_results(y_test, y_pred)
    display_results(y_test, y_pred)


train_and_display(1)