## Read logs.csv file and load data in pandas dataframe
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import matplotlib.pylab as pl
from mpl_toolkits.mplot3d import Axes3D


from sklearn.preprocessing import LabelEncoder

def combine_labels(df):
    df['combined_labels'] = 'sConesInit=' + df['sConesInit'].astype(str) + '_' + 'mConesInit=' + df['mConesInit'].astype(str) + '_' + 'maxProb' + df['maxProb'].astype(str) + '_' + 'cellRatio' + df['sCones_mCones_ratio'].astype(str)
    return df

def expand_df(df):
    ## Select only gridSize, totalCells, sCones_mCones_ratio, maxProb, sConesInit,mConesInit, VDRI[# of cells -> VDRI], NNRI[# of cells -> NNRI]
    df_filtered = df[['gridSize', 'totalCells', 'sCones_mCones_ratio', 'maxProb', 'sConesInit', 'mConesInit', 'VDRI[# of cells -> VDRI]', 'NNRI[# of cells -> NNRI]']]

    def expand_lists(row):
        vdri_list = row['VDRI[# of cells -> VDRI]']
        nnri_list = row['NNRI[# of cells -> NNRI]']

        expanded_row = {}
        # Add columns from the input row to the expanded row
        for col in row.index:
            if col not in ['VDRI[# of cells -> VDRI]', 'NNRI[# of cells -> NNRI]']:
                expanded_row[col] = row[col]

        expanded_row['VDRI[# of cells]'] = []
        expanded_row['VDRI'] = []
        expanded_row['NNRI[# of cells]'] = []
        expanded_row['NNRI'] = []

        for i in range(len(vdri_list)):
            expanded_row['VDRI[# of cells]'].append(vdri_list[i][0])
            expanded_row[f'VDRI'].append(float(vdri_list[i][1]))
            expanded_row[f'NNRI[# of cells]'].append(nnri_list[i][0])
            expanded_row[f'NNRI'].append(float(nnri_list[i][1]))

        return pd.Series(expanded_row)

    # Assuming your dataframe is named 'df'
    expanded_df = df_filtered.apply(expand_lists, axis=1)
    exploded_df = expanded_df.explode(['VDRI', 'NNRI', 'VDRI[# of cells]', 'NNRI[# of cells]'])
    return exploded_df

def encode_z_axis(df):
    label_encoder = LabelEncoder()

    # Combine the labels from the 4 columns into a single string column
    df['combined_labels'] = 'sConesInit=' + df['sConesInit'].astype(str) + '_' + 'mConesInit=' + df['mConesInit'].astype(str) + '_' + 'maxProb' + df['maxProb'].astype(str) + '_' + 'cellRatio' + df['sCones_mCones_ratio'].astype(str)

    # Encode the combined labels using the LabelEncoder
    df['z-axis'] = label_encoder.fit_transform(df['combined_labels'])
    return df


with open('output/logs.json', 'r') as f:
    data = json.load(f)

df = pd.DataFrame(data)


# Find the average and confidence interval of voronoi area variance across different combinations of sConesInit, mConesInit and maxProb
# Group by sConesInit, mConesInit and maxProb and calculate the mean and standard deviation of voronoi area variance
df_grouped = df.groupby(['sConesInit', 'mConesInit', 'maxProb', 'sCones_mCones_ratio'])\
    .agg(voronoi_area_variance_mean=('voronoi area variance', 'mean'),
         voronoi_area_variance_std=('voronoi area variance', 'std'),
         VDRIs=('VDRI[# of cells -> VDRI]', lambda x: sum(x, [])),
         NNRIs=('NNRI[# of cells -> NNRI]', lambda x: sum(x, []))).reset_index()

# Create a figure with one plot
plt.figure(figsize=(40,10))
df[df['sCones_mCones_ratio'] == 0.08].boxplot(column='voronoi area variance', by=['sConesInit', 'mConesInit', 'maxProb'])
plt.xticks(rotation=45, ha='right')  # Rotate labels and align them
plt.tight_layout()  # Adjust layout
plt.show()

######################

df_exploded = expand_df(df)
df_grouped_exploded = df_exploded.groupby(['sConesInit', 'mConesInit', 'maxProb', 'sCones_mCones_ratio', 'VDRI[# of cells]', 'NNRI[# of cells]']) \
    .agg(VDRI_mean=('VDRI', 'mean'),
         VDRI_std=('VDRI', 'std'),
         NNRI_mean=('NNRI', 'mean'),
         NNRI_std=('NNRI', 'std')).reset_index()
df_grouped_exploded.dropna(axis=0, subset=['VDRI_mean', 'NNRI_mean'], inplace=True)

df_exploded.dropna(axis=0, subset=['VDRI', 'NNRI'], inplace=True)
df_combined_labels = combine_labels(df_exploded)


df_encoded = encode_z_axis(df_grouped_exploded)

# Create a figure with one plot
x = df_encoded['VDRI[# of cells]']
y = df_encoded['VDRI_mean']
z = df_encoded['z-axis']

data = df_combined_labels.loc[:, ['VDRI[# of cells]', 'VDRI', 'NNRI[# of cells]', 'NNRI', 'combined_labels']]
# Substitute infinity with a very high number
data = data.replace([np.inf, -np.inf], 90000)


pl.figure()
ax = pl.subplot(projection='3d')
for i in range(len(df_encoded['z-axis'].unique())):
    y = df_encoded[df_encoded['z-axis'] == i]['VDRI[# of cells]']
    z = df_encoded[df_encoded['z-axis'] == i]['VDRI_mean']
    x = df_encoded[df_encoded['z-axis'] == i]['z-axis']
    ax.plot(x, y, z, label=i)

# Limit y-axis to 0-20
ax.set_zlim(0, 20)
pl.show()




# plt.figure(figsize=(40,10))
# df[df['sCones_mCones_ratio'] == 0.03].boxplot(column='voronoi area variance', by=['sConesInit', 'mConesInit', 'maxProb'])
# plt.xticks(rotation=45, ha='right')  # Rotate labels and align them
# plt.tight_layout()  # Adjust layout
# plt.show()



def plot_VDRI(data):
    y_values = {}

    for x, y in data:
        if y == 'NaN': # This is not ideal but just for now
            continue
        if x in y_values:
            y_values[x].append(y)
        else:
            y_values[x] = [y]

    # Calculate the average y-value for each x-value
    x_values = []
    y_averages = []
    for x, y_list in y_values.items():
        if 'Infinity' in y_list:
            y_avg = float('Infinity')
        else:
            y_avg = sum(y_list) / len(y_list)
        x_values.append(x)
        y_averages.append(y_avg)

    x, y = zip(*sorted(zip(x_values, y_averages)))

    plt.figure(figsize=(10, 6))
    plt.plot(x, y)
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.title('Data Plot')

    plt.legend()
    plt.grid(True)
    plt.show()

    return plt
