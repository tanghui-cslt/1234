import os
import numpy as np
import pandas as pd
from geneo.classifiers.utils import generate_latex_table

dirname = os.path.dirname(__file__)
os.path.join(dirname, 'knowledge_transfer')
subdirs = [os.path.join(path, f) for f in os.listdir(path)]
table_dict = {'dataset': []}

for subdir in subdirs:
    print("Reading {}".format(subdir))
    datasets = [os.path.join(subdir, f) for f in os.listdir(subdir)]
    geneo_column_name  = os.path.split(subdir)[-1]
    table_dict[geneo_column_name] = []
    table_dict["Glorot"] = []

    for dataset in datasets:
        print("\... Reading {}".format(dataset))
        histories = [np.load(os.path.join(dataset, f)).item() for f in os.listdir(dataset)
                     if 'history' in f]
        table_dict['dataset'].append(os.path.split(dataset)[-1])

        for history in histories:
            # loss = history['geneo_init']['loss']
            # val_loss = history['geneo_init']['val_loss']
            # acc = history['geneo_init']['acc']
            table_dict[geneo_column_name].append(history['geneo_init']['val_acc'][-1])
            table_dict["Glorot"].append(history['glorot_init']['val_acc'][-1])

d = pd.DataFrame(table_dict)
d = d.sort_values(by=['dataset'], ascending=False)

for c in d.columns[:4]:
    print(d[c].mean())

for dataset in np.unique(d.dataset):
    print("dataset ", dataset)
    for c in d.columns[:4]:
        print(c, "---")
        print("{} mean".format(dataset), d.loc[d['dataset']==dataset, c].mean())
        print("{} std".format(dataset), d.loc[d['dataset']==dataset, c].std())



d.to_latex(index=False)
