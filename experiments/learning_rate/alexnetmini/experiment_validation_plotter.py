import pymongo
import matplotlib.pyplot as plt
import numpy as np

# obtain runs collection created by sacred
db_client = pymongo.MongoClient()
sacred_db = db_client.get_database('sacred')
runs      = sacred_db.runs

# pipeline to get all accuracies with `_id`s
accuracies_pipeline = [
    {'$match': {'result': {'$ne': None}}},                   # filter broken experiments
    {'$match': {'config.base_lr': 0.2}},                     # use only lr=0.2
    {'$group': {'_id': '$config.schedule',                   # group schedule fn
                'accuracies': {'$addToSet': '$result'}}}     # make array from all accuracies
]
# make list, since the iterator is exhausted after one traversal
grouped_runs = list(sacred_db.runs.aggregate(accuracies_pipeline))
#
# run over the list of records/dicts once to group the values of each key (schedule_fn and
# accuracies) into separate lists so they can be used for boxplotting.
labels     = []
accuracies = []
for d in grouped_runs:
    labels.append(d['_id'])
    accuracies.append(d['accuracies'])

# show boxplots
f  = plt.figure()
ax = f.gca()
ax.boxplot(accuracies, labels=labels)

# plot the samples as dots with a random normal yoffset
for i in range(len(labels)):
    y = accuracies[i]
    x = np.random.normal(i + 1, 0.04, size=len(y))  # tick locations are [1, 2, 3, ...] for boxplots
    ax.plot(x, y, '.', alpha=0.3, markersize=20)

plt.show()
