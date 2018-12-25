import pymongo

import os

try:
    pwd = os.environ['MONGOPWD']
except KeyError:
    print('You need to set the MONGOPWD variable to connect to the database.')
    import sys
    sys.exit(1)

db_client = pymongo.MongoClient(f'mongodb://rasmus:{pwd}@35.189.247.219/sacred')
sacred_db = db_client.sacred
runs      = sacred_db.runs
metrics   = sacred_db.metrics


def get_metric_for_ids(name, ids, per_module=True):
    if per_module:
        metric = sacred_db.metrics.aggregate([
            {'$match': {'name': {'$regex': name}}},
            {'$match': {'run_id': {'$in': ids}}},
            {'$project': {'steps': True,
                          'values': True,
                          '_id': False,
                          'name': True}
             },
            {'$sort': {'name': 1}}
        ]
        )
    else:
        metric = sacred_db.metrics.aggregate([
            {'$match': {'name': {'$regex': name}}},
            {'$match': {'run_id': {'$in': ids}}},
            {'$project': {'steps': True,
                          'values': True,
                          '_id': False}
             }
        ]
        )

    return list(metric)


def prompt_delete(ids):
    '''Ask user if they __really__ want to delete.

    Parameters
    ----------
    ids :   list
            List of ids about to be deleted from `runs` along with their associated metrics.
    '''

    print(f'The following ids will be deleted: {ids}')
    print('DO NOT CALL THIS WHEN THERE ARE EXPERIMENTS RUNNING! Continue? ')
    if input('[yN]') != 'y':
        return False
    else:
        print('I warned you')
        return True


def delete_where(**kwargs):
    '''Delete runs and metrics where condition is met.

    Parameters
    ----------
    kwargs  :   dict
                The key-value pairs to be deleted. For conditioning based on dotted names (e.g.
                ``config.n_epochs = 30``), you can pass a dict directly:
                    ``delete_where(**{'config.n_epochs': 30})``
    '''
    pipeline = [{'$match': {k: v}} for k, v in kwargs.items()] + [{'$project': {'_id': True}}]
    ids_to_delete = list(map(lambda d: d['_id'], runs.aggregate(pipeline)))
    if prompt_delete(ids_to_delete):
        runs.delete_many({'_id': {'$in': ids_to_delete}})
        metrics.delete_many({'run_id': {'$in': ids_to_delete}})


def delete_invalid():
    '''Delete all experiments and associated metrics for which the result was ``None``'''

    pipeline = [
        {'$match': {'result': None}},
        {'$project': {'_id': True}},
    ]
    invalid_ids = list(map(lambda dickt: dickt['_id'], runs.aggregate(pipeline)))
    if not invalid_ids:
        print('Nothing to delete.')
    elif prompt_delete(invalid_ids):
        deleted_runs = runs.delete_many({'_id': {'$in': invalid_ids}})
        deleted_metrics = metrics.delete_many({'run_id': {'$in': invalid_ids}})

        print(f'Deleted {deleted_runs.deleted_count} runs and {deleted_metrics.deleted_count} metrics.')
