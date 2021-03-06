import sys
import pandas as pd
import numpy as np
import sqlite3
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    """
    Summary line.
    
    This function is to load the raw csv data
    
    Parameters:
    messages_filepath (str): path of disaster_message.csv
    categories_filepath (str): path of disaster_categories.csv
    
    Returns:
    DataFrame: merged disaster_message.csv and disaster_categories.csv
    
    """
    
    # load messages dataset
    messages = pd.read_csv(messages_filepath)
    
    # load categories dataset
    categories = pd.read_csv(categories_filepath)
    
    # merge datasets
    df = messages.merge(categories, how = 'outer', on = ['id'])
    
    return df
    

def clean_data(df):
    """
    Summary line.
    
    This function is to clean the data
    
    Parameters:
    df (DataFrame): merged disaster_message.csv and disaster_categories.csv
    
    Returns:
    Dataframe: cleaned data
    
    """
    
    # create a dataframe of the 36 individual category columns
    categories = df['categories'].str.split(';', expand = True)
    
    # use first row to rename new column names for categories
    row = categories.iloc[0]
    category_colnames = row.apply(lambda x : x[:-2])
    categories.columns = category_colnames
    
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].astype(str).str[-1]
    
        # convert column from string to numeric
        categories[column] = pd.to_numeric(categories[column])
        
    # set some categories to 1 where equals to 2 previously
    categories[categories['related'] == 2] = 1
    
    # drop the original categories column from `df`
    df = df.drop('categories', axis = 1)
    
    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], axis = 1)
    
    # drop duplicates
    df.drop_duplicates(inplace = True)
    
    return df

def save_data(df, database_filepath):
    """
    Summary line.
    
    This function is to save the data to database
    
    Parameters:
    df (DataFrame): cleaned data
    database_filename (str): 
    
    Returns:
    N/A
    
    """
    # save data to SQL
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    df.to_sql('DisasterResponseRawData', engine, if_exists='replace', index=False)  


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