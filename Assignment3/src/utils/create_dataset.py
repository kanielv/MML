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
        df = type

        train, test = train_test_split(df, random_state=15, train_size=0.7)

        train = train.reset_index(drop=True)
        test = test.reset_index(drop=True)

        for idx, i in train.iterrows():
            # print(row['filepath'], train_path + '/' + row['type'])
            shutil.copy(i['filepath'], train_path + '/' + i['type'])
        for idx, j in test.iterrows():
            shutil.copy(j['filepath'], test_path + '/' + j['type'])
        
        

        




