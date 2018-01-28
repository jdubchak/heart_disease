## Jordan Dubchak, Jan 2018
##
## split data into training and testing sets

## import libraries 
import pandas as pd
import numpy as np
from sklearn.model_selection import ShuffleSplit

## load in the data
combined = pd.read_csv("data/clean_data/heart_disease_cases.csv")

## get training set
combined_split = ShuffleSplit(n_splits=3, random_state=1234)
for train_index, test_index in combined_split.split(combined):
	train_index = train_index

## create empty data frames to append training and test values to 
train_df = pd.DataFrame(columns=combined.columns)
test_df = pd.DataFrame(columns=combined.columns)

## generate training and testing data 
for ind, row in combined.iterrows():
    if ind in train_index:
        train_df = train_df.append(row)
    else:
        test_df = test_df.append(row)

train_df = train_df.reset_index(drop=True)
test_df = test_df.reset_index(drop=True)

train_df.to_csv("data/clean_data/training_data.csv")
test_df.to_csv("data/clean_data/test_data.csv")