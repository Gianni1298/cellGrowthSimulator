## Read logs.csv file and load data in pandas dataframe
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('logs.csv', sep='|', on_bad_lines = 'warn',
                 names=['gridSize', 'sConesInit', 'mConesInit', 'sConesFinal', 'mConesFinal', 'maxProb', 'cell_indexes',
                        'blueHexCenters', 'voronoi areas', 'voronoi area variance', 'FTFreq-Magnitude', 'NN distances',
                        'ripleyG', 'ripleyF', 'ripleyJ', 'ripleyK', 'ripleyL', 'sCones-mCones ratio'], skiprows=1)

# Fill sCones-mCones ratio column with 0.08 if sCones-mCones ratio is null
df['sCones-mCones ratio'] = df['sCones-mCones ratio'].fillna(0.08)

# Find the average and confidence interval of voronoi area variance across different combinations of sConesInit, mConesInit and maxProb
# Group by sConesInit, mConesInit and maxProb and calculate the mean and standard deviation of voronoi area variance
df_grouped = df.groupby(['sConesInit', 'mConesInit', 'maxProb', 'sConesFinal', 'mConesFinal', 'sCones-mCones ratio'])\
    .agg({'voronoi area variance': ['mean', 'std']}).reset_index()

# Create a figure with one plot
plt.figure(figsize=(40,10))
df[df['sCones-mCones ratio'] == 0.08].boxplot(column='voronoi area variance', by=['sConesInit', 'mConesInit', 'maxProb'])
plt.xticks(rotation=45, ha='right')  # Rotate labels and align them
plt.tight_layout()  # Adjust layout
plt.show()

plt.figure(figsize=(40,10))
df[df['sCones-mCones ratio'] == 0.03].boxplot(column='voronoi area variance', by=['sConesInit', 'mConesInit', 'maxProb'])
plt.xticks(rotation=45, ha='right')  # Rotate labels and align them
plt.tight_layout()  # Adjust layout
plt.show()
