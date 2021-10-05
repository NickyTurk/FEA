from refactoring.utilities.multifilereader import MultiFileReader
import pickle

file_regex = 'NSGA_knapsack_3_objectives_ga_runs_'

stored_files = MultiFileReader(file_regex)
file_list = stored_files.path_to_files

parameters = [50, 100, 200]

for param in parameters:
    spread = 0
    HV = 0
    ND_size = 0
    amount = 0
    for file in file_list:
        if str(param) in file:
            #print(file)
            amount += 1
            obj = pickle.load(open(file, 'rb'))
            spread = spread + obj.iteration_stats[-1]['diversity']
            HV = HV + obj.iteration_stats[-1]['hypervolume']
            ND_size = ND_size + obj.iteration_stats[-1]['ND_size']
    HV = HV/amount
    ND_size = ND_size/amount
    spread = spread/amount
    print(param)
    print('hv: ', HV, 'nd: ', ND_size, 'spread: ', spread)

