'''
This file contains a bunch of utility functions for assignment 3
Author: Kaniel Vicencio
'''
import os
import csv
from tqdm import tqdm


def create_annotation_file(fp): 
    with open('annotations.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        field = ["type", "filename", "filepath" ]
        writer.writerow(field)

        for dir in tqdm(os.listdir(fp)):
            for idx, filename in enumerate(os.listdir(f"{fp}" + f"{dir}")): 
                row = [dir, filename, fp + dir + "/" + filename]
                writer.writerow(row)
    return None

