# import libraries
import pandas as pd
from sqlalchemy import create_engine
import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report
import pickle
import re
import sys


def load_data(database_filepath):
    '''
    Load Data from Database to a Dataframe and prepare for ML-Model
    IN: Database
    OUT: Dependent, Independent Variables and Category Names
    '''
    # load data from database
    engine = create_engine('sqlite:///'+database_filepath)
    df = pd.read_sql_table('Messages', engine)
    X = df.message.values
    y = df.drop(['id','original','message','genre'],axis=1).values 
    category_names  = df.drop(['id','original','message','genre'],axis=1).columns
    return X, y, category_names


def tokenize(text):
    '''
    Input: Message
    Output: Lower Tokenized Message excluding punctuation
    
    '''
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9]'," ",text)
    tokens = word_tokenize(text)
    
    return tokens



def build_model():
    '''
    Define Pipeline to preprocess and create estimators
    '''
    
    pipeline = Pipeline([
        ('vect', TfidfVectorizer(tokenizer=tokenize)),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])
    
    parameters = {'vect__stop_words': (None, 'english'),
              #'clf__estimator__n_estimators': [50, 100, 200],
              #'clf__estimator__min_samples_split': [2, 3, 4],                  
    }
    cv = GridSearchCV(pipeline, param_grid=parameters, verbose=3)
    
    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    '''
    Predict on new values and plot the metrics
    '''
    y_pred = model.predict(X_test)
    for cat in range(Y_test.shape[1]):
        print(classification_report(Y_test[:,cat], y_pred[:,cat]))
    pass


def save_model(model, model_filepath):
    '''
    Save model to model_filepath picke file
    '''
    pickle.dump(model.best_estimator_,open(model_filepath,"wb"))
    pass


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