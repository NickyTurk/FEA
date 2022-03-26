from sklearn.ensemble import RandomForestRegressor
from refactoring.utilities.field.create_prescription_datafile import create_pd_dataframe
from refactoring.optimizationproblems.prescription import Prescription
import pandas as pd
import numpy as np
import pickle, random, copy


class YieldPredictor:
    def __init__(self, field, agg_data_file, trained_model, data_headers, prescription=None):
        df = pd.read_csv(agg_data_file)
        self.headers = {c: i for i, c in enumerate(df.columns)}
        self.dps = df.to_numpy()
        self.nitrogen_dataframe = pd.DataFrame()
        self.data_headers = data_headers
        self.field = field
        self.gridcell_size = self.field.cell_list[0].gridcell_size/43560
        if prescription:
            new_df = create_pd_dataframe(prescription, field, self.headers, self.dps, transform_to_latlon=True)
            # self.data_headers.append('cell_index')
            self.nitrogen_dataframe = new_df.loc[:, data_headers]
        self.model = trained_model

    def calculate_yield(self, prescription=None):
        """
        Very convoluted right now because of the columns needed.
        Original data 'self.nitrogen_dataframe' needed for prediction does not use cell index,
        but we need the cell index for each data point to calculate the predicted yield for each cell.
        """
        nitrogen_dataframe = pd.DataFrame()
        if prescription:
            new_df = create_pd_dataframe(prescription, self.field, self.headers, self.dps)
            self.nitrogen_dataframe = new_df.loc[:, self.data_headers]
            adjusted_data_headers = copy.copy(self.data_headers)
            adjusted_data_headers.append('cell_index')
            nitrogen_dataframe = new_df.loc[:, adjusted_data_headers]
        yield_predictions = self.model.predict(self.nitrogen_dataframe)
        nitrogen_dataframe.loc[:, 'predicted'] = yield_predictions
        #print(nitrogen_dataframe.columns)
        actual_yield = 0
        for i in range(len(self.field.cell_list)):
            cell_predictions = nitrogen_dataframe.loc[nitrogen_dataframe['cell_index'] == i]
            actual_yield += np.mean(cell_predictions['predicted'])*self.gridcell_size
        return actual_yield


if __name__ == '__main__':
    agg_file = "C:/Users/f24n127/Documents/Work/Ag/Data/broyles_sec35mid_2016_yl_aggreg_20181112.csv"
    df = pd.read_csv(agg_file)
    y_labels = df['yl_2016']
    data_to_use = ['x', 'y', 'n_lbs_ac', 'elev_m', 'slope_deg', 'ndvi_2012', 'ndvi_2014', 'ndvi_2015', 'yl14_nn_bu_ac',
                 'n15_lbs_ac', 'n14_lbs_ac']
    x_data = df[data_to_use]
    rf = RandomForestRegressor()
    rf.fit(x_data, y_labels)

    field_file = "../utilities/saved_fields/sec35mid.pickle"
    field = pickle.load(open(field_file, 'rb'))
    field.fixed_costs = 1000
    random_global_variables = random.choices([80,100,120,140], k=len(field.cell_list))
    pr = Prescription(variables=random_global_variables, field=field)
    yp = YieldPredictor(prescription=pr, field=field, agg_data_file=agg_file, trained_model=rf, data_headers=data_to_use)
    yp.calculate_yield()


