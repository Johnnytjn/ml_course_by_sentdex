import numpy as np
import pandas as pd
from math import sqrt

df = pd.read_csv('breast-cancer-wisconsin.data.txt', index_col=0)
df.replace('?', -99999, inplace=True)
X = np.array(df.drop(['class'], 1))
y = np.array(df['class'])
