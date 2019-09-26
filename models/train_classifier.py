import sys
import numpy as np
import pandas as pd
import pickle
import sqlite3
import nltk
from sqlalchemy import create_engine
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import classification_report
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger'])


def load_data(database_filepath):
    """Returns features and targets as Numpy arrays (X,y). Category names as list"""
    engine = create_engine('sqlite:///' + str(database_filepath))
    df = pd.read_sql_query("select * from MessagesCategoriesClean;", engine)
    X = df['message'].values
    y = df.drop(['id', 'message', 'original', 'genre'],axis=1).values
    category_names = df.drop(['id', 'message', 'original', 'genre'],axis=1).columns.tolist()

    return X, y, category_names


def tokenize(text):
    """Converts messages into list of lemmatized words"""
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
    """Instantiates and returns instance of GridSearchCV as model. Pipeline is used to perform transformations before passing data to RandomForestClassifier"""
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier(n_jobs=-1)))
        ])
    parameters = {
        'clf__estimator__max_depth': [50, 75, 100],
        'clf__estimator__min_samples_split': [2, 5, 10],
        'clf__estimator__max_features': ['auto', 'sqrt'],
        'clf__estimator__min_samples_leaf': [1, 2, 4],
        'clf__estimator__n_estimators': [50, 75, 100],
        'clf__estimator__bootstrap': [True]
    }

    cv = GridSearchCV(pipeline, param_grid=parameters, n_jobs=-1, verbose=2)

    return cv


def evaluate_model(model, X_test, y_test, category_names):
    """Prints classification report for each category based on test data. Classification report includes precision, recall, f1-score, and support"""
    y_pred = model.predict(X_test)
    for i, column in enumerate(category_names):
        print(column)
        print(classification_report([y[i] for y in y_test], [y[i] for y in y_pred]))


def save_model(model, model_filepath):
    """Persists the model via pickling"""
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, y, category_names = load_data(database_filepath)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, y_train)
        best_model = model.best_estimator_

        print('Evaluating model...')
        evaluate_model(best_model, X_test, y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(best_model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()