import pandas as pd
from sqlalchemy import create_engine

# import data
def input_file(string):
    """
    Ask the user to input path of the data files
    :param string: the kind of data file, e.g messages
    :return: the dataframe read by pandas form the data file
    """
    print('Please enter the path of the {}:'.format(string))
    while True:
        try:
            file_path = input()
            df = pd.read_csv(file_path)
            print('Successfully read in {}!'.format(string))
            break
        except:
            print("file doesn't exist, please enter again:")
    return df


messages = input_file('messages file')
categories = input_file('categories file')

# merge two dataframe into one
df = messages.merge(categories, on='id')


def categories_col(text):
    """
    This function finds the categories needed
    :param text: string
    :return: list of string
    """
    categories = text.split(';')
    return [x[:-2] for x in categories]


categories = categories_col(df['categories'][0])


def getValues(text):
    """
    The function finds the values of the categories, 1 or 2 or 0, for one message
    :param text: string
    :return: list of int
    """
    categoriesValue = text.split(';')
    return [int(x[-1]) if (int(x[-1]) == 0 or int(x[-1]) == 1) else 1 for x in categoriesValue]


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
df_final.to_sql('DisasterResponse', engine, index=False, if_exists='replace')
