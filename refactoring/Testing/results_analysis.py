import pickle, glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

experiment_filenames = ["../../results/FEAMOO/CCEAMOO_Henrys_trial_3_objectives_linear_topo_ga_runs_100_population_500_0508121527.pickle",
                        "../../results/FEAMOO/CCEAMOO_Sec35Middle_trial_3_objectives_strip_topo_ga_runs_100_population_500_3007121518",
                        "../../results/FEAMOO/CCEAMOO_Sec35West_trial_3_objectives_strip_topo_ga_runs_100_population_500_0408143024.pickle",
                        "../../results/FEAMOO/NSGA2_Henrys_trial_3_objectives_ga_runs_200_population_500_2807110247.pickle",
                        "../../results/FEAMOO/NSGA2_Sec35Middle_trial_3_objectives_ga_runs_200_population_500_2807110338.pickle",
                        "../../results/FEAMOO/NSGA2_Sec35West_trial_3_objectives_ga_runs_200_population_500_2807110402.pickle",
                        "../../results/FEAMOO/FEAMOO_Sec35West_trial_3_objectives_strip_topo_ga_runs_100_population_500_0508090836.pickle",
                        "../../results/FEAMOO/",
                        "../../results/FEAMOO/"]
field_files= ["../utilities/saved_fields/Henrys.pickle","../utilities/saved_fields/sec35west.pickle","../utilities/saved_fields/sec35mid.pickle"]
field_names = ['henrys', 'sec35middle', 'sec35west']
methods = ["CCEAMOO", "NSGA2"] #, "FEAMOO"]
objectives = ["center", "fertilizer_rate", "jumps", "strat"]

all_data = dict() #{'henrys': { 'cceamoo':{'jumps': 0, 'strat': 0, 'fertilizer_rate': 0, 'center': 0}, 'nsga2': {} } }

for fieldfile, field_name in zip(field_files, field_names):
    field = pickle.load(open(fieldfile, 'rb'))
    all_data[field_name] = dict()
    for method in methods:
        all_data[field_name][method] = dict()
        experiment = [x for x in experiment_filenames if method in x and field_name in x.lower()]
        if experiment:
            experiment = experiment[0]
        else:
            break
        feamoo = pickle.load(open(experiment, 'rb'))
        print(feamoo.iteration_stats[-1])
        for obj in objectives:
            print(obj)
            to_find = '/home/amy/Documents/FEAMOO/' + method + '/' + method + "_" + field_name + '_prescription_' + obj + '*.csv'
            print(to_find)
            results = glob.glob(to_find)[0]
            results_df = pd.read_csv(results)
            nonzero_results = np.ma.masked_equal(results_df.iloc[:,-1:].to_numpy(), 0)
            avg_yld = np.mean(nonzero_results)
            all_data[field_name][method][obj] = avg_yld
            print(avg_yld)

print(all_data)

names = ['centr', 'fert', 'jumps', 'strat']

fig, axs = plt.subplots(1, 3, figsize=(9, 3), sharey=True)

ccea_yield = list(all_data['henrys']['CCEAMOO'].values())
nsga_yield = list(all_data['henrys']['NSGA2'].values())
#fea_yield = list(all_data['henrys']['FEAMOO'].values())

axs[0].scatter(names, ccea_yield)
axs[0].plot(names, ccea_yield)
axs[0].scatter(names, nsga_yield, color='#2ca02c', marker="^")
axs[0].plot(names, nsga_yield, color='#2ca02c')
axs[0].set_title("Henrys")

ccea_yield = list(all_data['sec35middle']['CCEAMOO'].values())
nsga_yield = list(all_data['sec35middle']['NSGA2'].values())
#fea_yield = list(all_data['sec35middle']['FEAMOO'].values())

axs[1].scatter(names, ccea_yield)
axs[1].plot(names, ccea_yield)
axs[1].scatter(names, nsga_yield, color='#2ca02c', marker="^")
axs[1].plot(names, nsga_yield, color='#2ca02c')
axs[1].set_title("Sec35Mid")

ccea_yield = list(all_data['sec35west']['CCEAMOO'].values())
nsga_yield = list(all_data['sec35west']['NSGA2'].values())
#fea_yield = list(all_data['sec35middle']['FEAMOO'].values())

axs[2].scatter(names, ccea_yield)
axs[2].plot(names, ccea_yield)
axs[2].scatter(names, nsga_yield, color='#2ca02c', marker="^")
axs[2].plot(names, nsga_yield, color='#2ca02c')
axs[2].set_title("Sec35West")
fig.suptitle('Categorical Plotting')

plt.show()