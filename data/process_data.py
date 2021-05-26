import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    '''
    Load messages and categories data to a DataFrame
    IN: Data filepaths (CSV)
    OUT: Dataframe
    '''
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = messages.merge(categories, on = 'id')
    return df


def clean_data(df):
    '''
    Clean Dataframe (df):
    - Separate categories text into different categories
    - Create new columns with categories names
    - Extract values from category names
    - Drop duplicates
    '''
    categories_df = df.categories.str.split(';',expand=True)
    row = categories_df.iloc[0]
    category_colnames = [cat[:-2] for cat in row]
    categories_df.columns = category_colnames
    for column in categories_df:
        # set each value to be the last character of the string
        categories_df[column] = categories_df[column].str[-1] 
        # convert column from string to numeric
        categories_df[column].astype(int)
    df.drop(['categories'],axis=1,inplace=True)
    df = pd.concat([df,categories_df],axis=1)
    df.drop_duplicates(inplace=True)
    #df = df[df.related.isnull() == False]
    return df


def save_data(df, database_filename):
    '''
    Save cleaned Dataframe(df) to SQLlite Database (database_filename)
    '''
    engine = create_engine('sqlite:///'+database_filename)
    df.to_sql('Messages', engine, index=False,if_exists='replace')
    pass  


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