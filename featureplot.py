import numpy as np
import pandas as pd 
import seaborn as sns 
import matplotlib.pyplot as plt 

df = pd.read_csv('/Users/vantran/Desktop/AAAAAA/resultsparallel/accuracy_bag.csv')
Features = df['Risk Factors']
Probability = df['Probability']

print(Probability)

