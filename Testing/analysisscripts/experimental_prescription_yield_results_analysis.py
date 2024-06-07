import math
import pickle, glob
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np

from predictionalgorithms.yieldprediction import create_indexed_dataframe

field_files = ["../../utilities/saved_fields/sec35mid.pickle"]
field_names = ["Sec35Mid"]
methods = ["CCEAMOO", "NSGA2", "FEAMOO"]
objectives = ["center", "fertilizer_rate", "jumps", "strat"]

all_data = (
    dict()
)  # {'henrys': { 'cceamoo':{'jumps': 0, 'strat': 0, 'fertilizer_rate': 0, 'center': 0}, 'nsga2': {} } }
for fieldfile, field_name in zip(field_files, field_names):
    field = pickle.load(open(fieldfile, "rb"))
    all_data[field_name] = dict()
    for method in methods:
        all_data[field_name][method] = dict()
        for obj in objectives:
            to_find = (
                "D:/Prescriptions/CNN_predictions/"
                + "/"
                + method
                + "_"
                + field_name
                + "_prescription_"
                + obj
                + "_objective_runs*.csv"
            )
            results = glob.glob(to_find)[0]
            if not results:
                to_find = (
                    "D:/Prescriptions/CNN_predictions/"
                    + field_name
                    + "_"
                    + method
                    + "_prescription_"
                    + obj
                    + "*.csv"
                )
                results = glob.glob(to_find)[0]
            print(results)
            results_df = pd.read_csv(results)
            new_df = create_indexed_dataframe(results_df, field, transform_to_latlon=True)
            applicator = 0
            yld = 0
            for i, cell in enumerate(field.cell_list):
                point_in_cell = new_df.loc[new_df["cell_index"] == i]
                if np.array(point_in_cell).size > 0:
                    non_zero = np.array([x for x in point_in_cell[12] if not math.isnan(x)])
                    if non_zero.size > 0:
                        yld = yld + np.mean(non_zero) * 0.98
                        applicator = (
                            applicator + np.mean(point_in_cell[point_in_cell.columns[-3]]) * 0.98
                        )
            net_return = yld * 5.30 - (applicator * 1.05) - field.fixed_costs
            print("NR: ", net_return)

            # nonzero_results = np.ma.masked_equal(results_df.iloc[:,-1:].to_numpy(), 0)
            # avg_yld = np.mean(nonzero_results)
            # all_data[field_name][method][obj] = avg_yld
            # print(field_name, ' ', method, ' ', obj, ' yield: ', str(avg_yld))

names = ["centr", "fert", "jumps", "strat"]

fig, axs = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)
axs[0].set_ylabel("bushels of wheat/acre")

ccea_yield = list(all_data["henrys"]["CCEAMOO"].values())
nsga_yield = list(all_data["henrys"]["NSGA"].values())
fea_yield = list(all_data["henrys"]["FEAMOO"].values())
comb_yield = list(all_data["henrys"]["COMBINED"].values())

axs[0].scatter(names, comb_yield, color="black", marker="s")
axs[0].plot(names, comb_yield, color="black")
axs[0].scatter(names, ccea_yield)
axs[0].plot(names, ccea_yield)
axs[0].scatter(names, nsga_yield, color="#2ca02c", marker="^")
axs[0].plot(names, nsga_yield, color="#2ca02c")
axs[0].scatter(names, fea_yield, color="tab:red", marker="*")
axs[0].plot(names, fea_yield, color="tab:red")
axs[0].set_title("Henrys")

ccea_yield = list(all_data["sec35middle"]["CCEAMOO"].values())
nsga_yield = list(all_data["sec35middle"]["NSGA"].values())
fea_yield = list(all_data["sec35middle"]["FEAMOO"].values())
comb_yield = list(all_data["sec35middle"]["COMBINED"].values())


axs[1].scatter(names, comb_yield, color="black", marker="s")
axs[1].plot(names, comb_yield, color="black")
axs[1].scatter(names, ccea_yield)
axs[1].plot(names, ccea_yield)
axs[1].scatter(names, nsga_yield, color="#2ca02c", marker="^")
axs[1].plot(names, nsga_yield, color="#2ca02c")
axs[1].scatter(names, fea_yield, color="tab:red", marker="*")
axs[1].plot(names, fea_yield, color="tab:red")
axs[1].set_title("Sec35Mid")

ccea_yield = list(all_data["sec35west"]["CCEAMOO"].values())
nsga_yield = list(all_data["sec35west"]["NSGA"].values())
fea_yield = list(all_data["sec35west"]["FEAMOO"].values())
comb_yield = list(all_data["sec35west"]["COMBINED"].values())


axs[2].scatter(names, comb_yield, color="black", marker="s")
axs[2].plot(names, comb_yield, color="black")
axs[2].scatter(names, ccea_yield)
axs[2].plot(names, ccea_yield)
axs[2].scatter(names, nsga_yield, color="#2ca02c", marker="^")
axs[2].plot(names, nsga_yield, color="#2ca02c")
axs[2].scatter(names, fea_yield, color="tab:red", marker="*")
axs[2].plot(names, fea_yield, color="tab:red")
axs[2].set_title("Sec35West")
fig.suptitle("")

ccea = mlines.Line2D([], [], color="#1f77b4", marker=".", markersize=12, label="CC-NSGA-II")
nsga = mlines.Line2D([], [], color="#2ca02c", marker="^", markersize=12, label="NSGA-II")
fea = mlines.Line2D([], [], color="tab:red", marker="*", markersize=12, label="F-NSGA-II")
comb = mlines.Line2D([], [], color="black", marker="s", markersize=10, label="Union")
axs[1].legend(
    handles=[nsga, ccea, fea, comb],
    bbox_to_anchor=(0, 1.15, 1.0, 0.105),
    loc="center",
    ncol=4,
    mode="expand",
    borderaxespad=0.0,
)
# plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
plt.suptitle("Predicted Yield", size="18")
fig.tight_layout(pad=1)
plt.show()
