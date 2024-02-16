import pandas as pd
from difflib import SequenceMatcher
# from fuzzywuzzy import fuzz


# Function to calculate similarity between two strings
def similar(a, b):
    # print("1")
    return SequenceMatcher(None, a, b).ratio()
# Function to find matching rows based on text similarity
def find_matching_rows(df1, df2, threshold=0.7):
    matching_rows = []

    for index1, row1 in df1.iterrows():
        for index2, row2 in df2.iterrows():
            similarity_score = similar(row1['text'], row2['text'])
            if similarity_score >= threshold:
                matching_rows.append((index1, index2, similarity_score))
    # print("2")
    return matching_rows

# Function to create the final DataFrame
def create_final_dataframe(df1, df2, matching_rows):
    final_data = []

    for index1, index2, similarity_score in matching_rows:
        row1_data = df1.loc[index1].to_dict()
        row2_data = df2.loc[index2].to_dict()

        final_data.append({**row1_data, **row2_data})
    # print("3")
    return pd.DataFrame(final_data)

# Read the CSV files
# file1_path = 'file1.csv'
# file2_path = 'file2.csv'

# df1 = pd.read_csv(file1_path)
# df2 = pd.read_csv(file2_path)

# Find matching rows
# matching_rows = find_matching_rows(df1, df2)

# # Create the final DataFrame
# final_df = create_final_dataframe(df1, df2, matching_rows)

# # Print the three DataFrames
# print("DataFrame 1:")
# print(df1)

# print("\nDataFrame 2:")
# print(df2)

# print("\nFinal DataFrame:")
# print(final_df)

# Save the final DataFrame to a CSV file
# final_df.to_csv('final_result.csv', index=False)