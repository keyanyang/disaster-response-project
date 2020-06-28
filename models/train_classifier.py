import sys
import pickle
import pandas as pd
import numpy as np
from sqlalchemy import create_engine

import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
nltk.download(['punkt', 'wordnet'])


from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import classification_report, accuracy_score


def load_data(database_filepath):
    """Load data from SQLite database and split it into predictor variable, response variables and category names.

    Parameters
    ----------
    database_filepath : str
        The file location of the SQLite database

    Returns
    -------
    X: Series
        predictor variablbe as feature
    Y: DataFrame
        response variables as target
    category_names: list
        list of column names of target
    """
    # load data from database
    engine = create_engine(f'sqlite:///{database_filepath}')
    df = pd.read_sql_table('Message', engine)
    X = df['message'].values
    Y = df[list(df)[4:]]
    category_names = Y.columns
    return X, Y, category_names


def tokenize(text):
    """Apply case normalization, lemmatize and tokenize text.

    Parameters
    ----------
    text : str
        Text in source format

    Returns
    -------
    list
        a cleaned tokenized list
    """
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
    """Builds classification model.

    Parameters
    ----------
    None

    Returns
    -------
    Object
        a multi-output classification model
    """
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])

    parameters = {
        # 'tfidf__use_idf': (True, False),
        # 'vect__ngram_range': ((1, 1), (1, 2)),
        'clf__estimator__n_estimators': [50, 100],
        # 'clf__estimator__min_samples_split': [2, 4],
    }

    cv = GridSearchCV(pipeline, param_grid=parameters, verbose=2, n_jobs=-1)
    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    """Evaluate the model against a test dataset.

    Parameters
    ----------
    text : str
        Text in source format

    Returns
    -------
    list
        a cleaned tokenized list
    """
    y_pred = model.predict(X_test)
    # check the second warning from https://scikit-learn.org/stable/modules/multiclass.html
    # print(classification_report(y_pred, Y_test.values, target_names=category_names))
    for i, col in enumerate(category_names):
        print(f"Accuracy scores for {col} is {accuracy_score(Y_test.values[:,i], y_pred[:,i])}")
    print('-' * 50)
    print(f'Mean Accuracy Score: {np.mean(Y_test.values == y_pred)}')


def save_model(model, model_filepath):
    """Save the model as a pickle file.

    Parameters
    ----------
    model : Object
        Trained model to be saved
    model_filepath: str
        The location of the pickle file

    Returns
    -------
    list
        a cleaned tokenized list
    """
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