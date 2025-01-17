import sys
import numpy as np
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """Converts message and category CSVs into Pandas dataframes, merges into one, then returns dataframe"""
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = messages.merge(categories, on='id')
    return df

def clean_data(df):
    """Expands semicolon delimited categories into their own df columns and slices values to return only 1 or 0"""
    categories = df['categories'].str.split(pat=';', expand=True)

    row = categories.loc[0]
    category_colnames = row.apply(lambda x: x[:-2])
    categories.columns = category_colnames
    for column in categories:
        # sets each value to be the last character of the string (0 or 1)
        categories[column] = categories[column].apply(lambda x: x[-1])
        categories[column] = pd.to_numeric(categories[column])

    df = df.drop('categories', axis=1)
    df2 = pd.concat([df, categories], axis=1)
    df2 = df2.drop_duplicates()
    return df2
    
def save_data(df, database_filename):
    """Saves dataframe of messages and categories to SQLite database"""
    engine = create_engine('sqlite:///data/DisasterResponse.db')
    df.to_sql('MessagesCategoriesClean', engine, index=False)


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