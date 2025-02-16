# <YOUR_IMPORTS>
import dill
import pandas as pd
import os
import glob
from datetime import datetime
import json
from os import listdir
from os.path import isfile, join
import logging


#path = os.environ.get('PROJECT_PATH', '.')
path = 'airflow_hw'
result_data = pd.DataFrame(columns = ['id', 'predict'])


def predict(model):
    path_files = path + '/data/test/*json'
    for json_files_path in glob.iglob(path_files):
        with open(json_files_path) as fin:
            form = json.load(fin)
            df = pd.DataFrame.from_dict([form])
            pred = model.predict(df)
            result_data.loc[len(result_data.index)] = [str(df['id']).split(' ')[4].split('\n')[0], pred[0]]
    result_data.to_csv(f'{path}/data/predictions/preds_{datetime.now().strftime("%Y%m%d%H%M")}.csv', index=False)
    print(result_data)


if __name__ == '__main__':
    latest_model = sorted(os.listdir(f'{path}/data/models'))[-1]
    with open(f'{path}/data/models/{latest_model}', 'rb') as file:
        model = dill.load(file)
    predict(model)