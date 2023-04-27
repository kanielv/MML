'''
This file contains a bunch of utility functions for assignment 3
Author: Kaniel Vicencio
'''
import os
import csv
from tqdm import tqdm


def create_annotation_file(fp): 
    with open('../dataset/annotations.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        field = ["type", "filename", "filepath" ]
        writer.writerow(field)

        for dir in tqdm(os.listdir(fp)):
            for idx, filename in enumerate(os.listdir(f"{fp}" + f"{dir}")): 
                row = [dir, filename, fp + dir + "/" + filename]
                writer.writerow(row)
    return None

import pandas as pd
def get_annotations(csv_path):
    return pd.read_csv(csv_path)

def get_annotations_types(df):
    
    df_angry = df[df['type'] == 'angry']
    df_fear = df[df['type'] == 'fear']
    df_happy = df[df['type'] == 'happy']
    df_sad = df[df['type'] == 'sad']

    return [df_angry, df_fear, df_happy, df_sad]
    

from sklearn.model_selection import train_test_split
import shutil   
def split_data(csv_path):
    df = get_annotations(csv_path)

    #Get types as data frames 
    types = get_annotations_types(df)
    
    train_path = "../dataset/train"
    test_path = "../dataset/test"
    for type in types:
        train, test = train_test_split(type, shuffle=True, train_size=0.7)
        train = type.reset_index()
        test = type.reset_index()
        
        print(train.head())

        # for idx, row in train.iterrows():
        #     # print(row['filepath'], train_path + '/' + row['type'])
        #     shutil.copy(row['filepath'], train_path + '/' + row['type'])
        # for idx, row in test.iterrows():
        #     shutil.copy(row['filepath'], test_path + '/' + row['type'])
        
        

        




