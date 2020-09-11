import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import glob

path = r'C:/Users/f24n127/Documents/School/FEA/pso/results'
all_files = glob.glob(path + "/F*_diff_grouping_small_epsilon.csv")

li = []

for filename in all_files:
    df = pd.read_csv(filename, index_col=None, header=0)
    li.append(df)

frame = pd.concat(li, axis=0, ignore_index=True)
print(len(li))
print(frame)



# data = np.concatenate((spread, center, flier_high, flier_low))
# fig1, ax1 = plt.subplots()
# ax1.set_title('Basic Plot')
# ax1.boxplot(data)