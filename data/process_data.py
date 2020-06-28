import sys
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    """Load messages and categories from csv files into a pandas dataframe.

    Parameters
    ----------
    messages_filepath : str
        The file location of the messages csv file
    categories_filepath : str
        The file location of the categories csv files

    Returns
    -------
    DataFrame
        a pandas dataframe including messages and categories data
    """
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = messages.merge(categories, on='id')
    return df


def clean_data(df):
    """Clean data by converting the category column into multiple boolean columns.

    Parameters
    ----------
    df : DataFrame
        A pandas dataframe including original messages and categories data

    Returns
    -------
    DataFrame
        Cleaned pandas dataframe including messages and categories data
    """
    # create a dataframe of the 36 individual category columns
    categories = df['categories'].str.split(';', expand=True)
    # select the first row of the categories dataframe
    row = categories.loc[0,:]
    category_colnames = row.str.split('-').str.get(0)
    # rename the columns of `categories`
    categories.columns = category_colnames

    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].apply(lambda x: x[-1:])
        
        # convert column from string to numeric
        categories[column] = pd.to_numeric(categories[column])
    
    # drop the original categories column from `df`
    df.drop('categories', axis=1, inplace=True)
    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], axis=1)
    # drop duplicates
    df.drop_duplicates(inplace=True)
    return df


def save_data(df, database_filename):
    """Save data from a pandas dataframe into SQLite database.

    Parameters
    ----------
    df : DataFrame
        A cleaned pandas dataframe including messages and categories data
    database_filename: str
        The file location of database output

    Returns
    -------
    None
    """
    engine = create_engine(f'sqlite:///{database_filename}')
    df.to_sql('Message', engine, index=False)


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()