from tqdm import tqdm
import numpy as np
import pandas as pd
import multiprocessing

from os import listdir
from os.path import isfile, join


path_name = './'
file_names = [path_name+f for f in listdir(path_name) if isfile(join(path_name, f))]

for file in file_names:
    print(file)
    df = pd.read_excel(file, sheet_name=-1)
    # print(df)


