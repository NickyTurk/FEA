import bnlearn, pgmpy

DAG = bnlearn.import_DAG('sprinkler', CPD=True)

df = bnlearn.sampling(DAG, n=1000)

print(df.head())