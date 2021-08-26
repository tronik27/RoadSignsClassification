import pandas as pd


train_path = 'D:/MIFI/SCIENTIFIC WORK/DATASETS/German road signs classification/Train.csv'
test_path = 'D:/MIFI/SCIENTIFIC WORK/DATASETS/German road signs classification/Test.csv'
train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)
print(train_df['ClassId'].nunique(), train_df['ClassId'].nunique())
print(train_df.head(20))
print(train_df['Width'].median(), train_df['Height'].median())
print(len(train_df), train_df['ClassId'].min(), train_df['ClassId'].max())
