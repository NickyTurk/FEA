from utilities.multifilereader import MultiFileReader
import pickle, random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

algorithms = ['NSGA2', 'NSGA3']  # MOEAD, SPEA2
obj = 5
problems = ['DTLZ5', 'DTLZ6', 'WFG3', 'WFG7']  # ['DTLZ1', 'DTLZ2', 'DTLZ3', 'DTLZ4', 'DTLZ5', 'DTLZ6'] #, 'WFG1', 'WFG2', 'WFG3', 'WFG4', 'WFG5', 'WFG7']
comparing = [['NSGA2', 'k_05_l_02'], ['NSGA2', 'k_05_l_03'],['NSGA2', 'k_05_l_04'],['NSGA2', 'k_05_l_05'],
             ['NSGA2', 'k_04_l_02'], ['NSGA2', 'k_04_l_03'],['NSGA2', 'k_04_l_04'],['NSGA2', 'k_04_l_05'],
             ['NSGA2', 'k_025_l_02'], ['NSGA2', 'k_025_l_03'],['NSGA2', 'k_025_l_04'],['NSGA2', 'k_025_l_05']]
markers = ["-", ":", "--", "-.", (0, (1, 10)), (0, (1, 1)), (5, (10, 3)), (0, (5, 10)), (0, (5, 1)), (0, (3, 10, 1, 10)), (0, (3, 5, 1, 5, 1, 5)), (0, (3, 1, 1, 1))]

fig = plt.figure()
for i,problem in enumerate(problems):
    file_regex = r'_' + problem + r'_(.*)' + str(obj) + r'_objectives_'
    stored_files = MultiFileReader(file_regex,
                                   dir="C:\\Users\\amy_l\\PycharmProjects\\FEA\\results\\factorarchive\\full_solution\\" + problem + "\\")
    experiment_filenames = stored_files.path_to_files
    plt.subplot(2,2,i+1)
    for j,compare in enumerate(comparing):
        experiments = [x for x in experiment_filenames if compare[0] in x and compare[1] in x]  # and 'PBI' not in x
        HV = []
        spread = []
        if experiments:
            rand_int = random.randint(0, len(experiments) - 1)
            for experiment in experiments:
                try:
                    results = pickle.load(open(experiment, 'rb'))
                except EOFError:
                    print('issues with file: ', experiment)
                    continue
                HV.append([res['hypervolume'] for res in results.iteration_stats])
                spread.append([res['diversity'] for res in results.iteration_stats])
        else:
            break
        points_to_plot_HV = np.average(HV, axis=0)
        if problem == "WFG7":
            points_to_plot_HV = points_to_plot_HV*10
        points_to_plot_spread = np.average(spread, axis=0)
        if problem == "DTLZ5":
            points_to_plot_spread = points_to_plot_spread * 10
        if problem == "WFG3":
            points_to_plot_spread = points_to_plot_spread * 100
        x_points = [x for x in range(99)]

        # plot with x and y data
        #plt.plot(x_points, points, _to_plot_HV, label=compare[1])

        # # plot with x and y data
        plt.plot(x_points, points_to_plot_spread, linestyle=markers[j], label=compare[1])
        plt.title(problem + " " + str(obj) + " objectives")

plt.suptitle('Spread Indicator')
lines = []
labels = []

for ax in fig.axes:
    Line, Label = ax.get_legend_handles_labels()
    # print(Label)
    lines.extend(Line)
    for label in Label:
        label = label.replace('k_', 'k: ')
        label = label.replace('l_', 'l: ')
        label = label.replace('_', ', ')
        label = label.replace("0", "0.")
        labels.append(label)
    break

fig.legend(lines, labels=labels,
           loc="center right")
plt.subplots_adjust(left=0.1,
                    bottom=0.1,
                    right=0.75,
                    top=0.85,
                    wspace=0.4,
                    hspace=0.4)
plt.show()
