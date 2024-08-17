import pandas as pd 

def add_number_of_characters_column(data_frame):
    """
    Adds a new column to the given DataFrame that contains the number of characters in each label.

    This function takes in a DataFrame as an argument and adds a new column to it. The new column is named
    'number_of_characters' and contains the number of characters in each label in the 'label' column of the
    DataFrame.

    Args:
        data_frame (pandas.DataFrame): A DataFrame containing a 'label' column.

    Returns:
        pandas.DataFrame: A new DataFrame with an additional column containing the number of characters in each label.
    """

    # Initialize a list to store the number of characters in each label
    number_of_characters = []

    # Loop through each label in the DataFrame
    for label in data_frame['label']:
        # Append the length of the label to the list
        number_of_characters.append(len(label))

    # Add a new column to the DataFrame called 'number_of_characters' and set its values to the list of
    # number of characters
    data_frame['number_of_characters'] = number_of_characters

    # Return the updated DataFrame with the new column
    return data_frame