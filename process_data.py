import pandas as pd
from sqlalchemy import create_engine


# import data
messages = pd.read_csv('messages.csv')
categories = pd.read_csv('categories.csv')

# merge two dataframe into one
df = messages.merge(categories, on='id')


def categories_col(text):
    '''
    This function finds the categories needed
    :param text: string
    :return: list of string
    '''
    categories = text.split(';')
    return [x[:-2] for x in categories]

categories = categories_col(df['categories'][0])



def getValues(text):
    '''
    The function finds the values of the categories, 1 or 2 or 0, for one message
    :param text: string
    :return: list of int
    '''
    categoriesValue = text.split(';')
    return [int(x[-1]) for x in categoriesValue]

# get the values of all the messages and create a value dataframe
all_values = []
for values in df['categories']:
    all_values.append(getValues(values))

new_df = pd.DataFrame(data=all_values, columns=categories)

new_df = new_df.reset_index()
new_df.drop(columns=['index'], inplace=True)


# combine the value dataframe with the original dataframe and remove the categories column
df_final = pd.concat([df, new_df], axis=1)
df_final.drop(columns=['categories'], inplace=True)
# print(df_final.shape)

# drop duplicates row
df_final = df_final.drop_duplicates()

# save the dataframe to the sql database
engine = create_engine('sqlite:///DisasterResponse.db')
df_final.to_sql('DisasterResponse', engine, index=False)