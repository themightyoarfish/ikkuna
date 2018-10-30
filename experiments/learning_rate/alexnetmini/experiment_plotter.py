import pymongo
import matplotlib.pyplot as plt

# obtain runs collection created by sacred
db_client = pymongo.MongoClient()
sacred_db = db_client.get_database('sacred')
runs      = sacred_db.runs

# pipeline to get all accuracies with `_id`s
accuracies_pipeline = pipeline = [{'$match': {'config.base_lr': 0.5}},              # use only lr=0.5
                                  {'$group': {'_id': '$config.schedule',            # group by id, use schedule fn as name
                                              'acc' : {'$addToSet': '$result'}}}]   # make array from all accuracies
grouped_runs = sacred_db.runs.aggregate(accuracies_pipeline)

# run over the list of records/dicts once to group the values of each key (schedule_fn and
# accuracies) into separate lists so they can be used for boxplotting.
labels     = []
accuracies = []
for d in grouped_runs:
    labels.append(d['_id'])
    accuracies.append(d['acc'])

# show boxplots
f  = plt.figure()
ax = f.gca()
ax.boxplot(accuracies, labels=labels)
plt.show()
