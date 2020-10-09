import pandas as pd
import re
import matplotlib.pyplot as plt
import os

# Fac is string of format [(0,1,2),(3,4),(4,5,6,7)]
# Turn this into list of tuples
def parse_factors(fac):
    tups = re.findall("(?:\([^)]+\)\s*)+", fac)  # gets the tuples copied from internet https://stackoverflow.com/questions/51965798/python-regular-expressions-extract-a-list-of-tuples-after-a-keyword-from-a-text
    # tups is list of form ['(a,b,c)', '(d,e,f)']
    l = [] # list of tuples
    for t in tups:
        tup = map(int, t.strip('(),').split(',')) # get rid of parenthesis and split, and applies int function
        tup = tuple(tup)
        l.append(tup)
    return l

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def parse_file(file):
    filename = "results\\" + file + "_m4_diff_grouping_small_epsilon.csv"

    dataframe = pd.read_csv(filename)
    # print(dataframe[2])
    # print(dataframe['DIMENSION'])

    small_ep = dataframe.loc[dataframe['EPSILON'] == min(dataframe['EPSILON'])]
    for indx, row in small_ep.iterrows():
        factors = row['FACTORS']
        groups = parse_factors(factors)
        num_factors = int(row['DIMENSION'])

        group_nums = [[] for _ in range(num_factors)]  # make 2d list for each variable
        for var in range(num_factors): # go through variables
            for i in range(len(groups)):  # go through factors
                if var in groups[i]:  # check if the variable is in the factor
                    group_nums[var].append(i + 1)

        # now need to flatten into (x, y) points
        x = []
        y = []
        for var in range(len(group_nums)):
            for g in group_nums[var]:
                x.append(var)
                y.append(g)

        plt.scatter(x, y)
        plt.title("Group Structure")
        plt.xlabel('Variable')
        plt.ylabel('Group')

        dir = "results\\plots\\" + file
        mkdir(dir)
        plt.savefig(dir + "\\dim_" + str(num_factors) + ".png")

        plt.show()


        # save as csv as well
        merge = [(x[i], y[i]) for i in range(len(x))]
        csv_out = "Variable, Group"
        for x, y in merge:
            csv_out += str(x) + "," + str(y) + '\n'

        file_out = open(dir + "\\dim_" + str(num_factors) + ".csv", 'w')
        file_out.write(csv_out)
        file_out.close()


for i in range(20):
    s = "F" + str(i + 1)
    print(s)
    parse_file(s)

