import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import glob
from ast import literal_eval

path = r'C:/Users/f24n127/Documents/School/FEA/pso/results'
all_files = glob.glob(path + "/F*_m4_diff_grouping_small_epsilon.csv")

li = []

for filename in all_files:
    df = pd.read_csv(filename, index_col=None, header=0)
    li.append(df)

frame = pd.concat(li, axis=0, ignore_index=True)

epsilons = frame.EPSILON.unique()
functions = frame.FUNCTION.unique()
dimensions = frame.DIMENSION.unique()

print(functions)

sum_nr_group_epsilon_dimension = np.zeros([len(epsilons), len(dimensions)])

for i, row in frame.iterrows():
    e = np.where(epsilons == row['EPSILON'])
    d = np.where(dimensions == row['DIMENSION'])
    sum_nr_group_epsilon_dimension[e, d] += row['NR_GROUPS']

group_size_epsilon_dimension = np.zeros([len(epsilons), len(dimensions), len(functions)])
for i, row in frame.iterrows():
    e = np.where(epsilons == row['EPSILON'])
    d = np.where(dimensions == row['DIMENSION'])
    f = np.where(functions == row['FUNCTION'])
    factors = literal_eval(row['FACTORS'])
    sum_len = 0
    for fac in factors:
        sum_len += len(fac)
    group_size_epsilon_dimension[e, d, f] = (sum_len/len(factors))

# Create a figure instance
fig, axes = plt.subplots(ncols=2, nrows=2, sharey=True, sharex=True)
a = plt.gca()
a.axes.yaxis.set_ticks([])
a.axes.xaxis.set_ticks([])
for i, eps in enumerate(group_size_epsilon_dimension):
    print(eps.shape)
    # Create an axes instance
    ax = fig.add_subplot(2, 2, i+1)
    ax.set_title('Epsilon value: '+ str(epsilons[i]))
    # Create the boxplot
    bp = ax.boxplot(np.transpose(eps))
plt.gcf().text(0.45, 0.01, 'Dimensions', fontsize=12)
plt.gcf().text(0.01, 0.45, 'Group Size', fontsize=12, rotation=90)
fig.suptitle('Boxplots of group size across 20 functions')
plt.show()
# plt.savefig('boxplot_groupsize')