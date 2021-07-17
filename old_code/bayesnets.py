from pgmpy.sampling import BayesianModelSampling
from pgmpy.models.BayesianModel import BayesianModel
from pgmpy.readwrite import BIFReader
import pyreadr

reader = BIFReader("../BN_data/hailfinder.bif")
DAG = reader.get_model()


#mb = DAG.get_markov_blanket('Age')
inference = BayesianModelSampling(DAG)
df = inference.forward_sample(size=100)  # Size = size of sample to be generated, return pd df
print(df.dtypes)