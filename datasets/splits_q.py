import os
import pandas as pd


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
name = 'DILIconcern'
data = pd.read_csv(os.path.join(BASE_DIR, '{}/raw/{}.csv'.format(name, name)))
data_1 = data[data['diliconcern'] == 0]
data_2 = data[data['diliconcern'] == 1]
data_3 = data[data['diliconcern'] == 2]
query_index = []
for ls in [data_1, data_2, data_3]:
    ids = ls.sample(80).index
    for id in ids:
        query_index.append(id)

suport_index = [id for id in range(len(data)) if id not in query_index]
query_data = data.iloc[query_index]
support_data = data.iloc[suport_index]
query_data.to_csv(os.path.join(BASE_DIR, '{}_query/raw/{}_query.csv'.format(name, name)), index=False)
support_data.to_csv(os.path.join(BASE_DIR, '{}_support/raw/{}_support.csv'.format(name, name)), index=False)
