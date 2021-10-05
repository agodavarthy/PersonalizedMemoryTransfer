import pandas as pd
import numpy as np
import portalocker
import os
import copy
import datetime as dt
from termcolor import colored
import pymongo
import atexit
from typing import List, Union


class MongoDBTracker(object):

    def __init__(self, unique_keys: list, db_name: str = 'results',
                 collection_name: str = 'convmovie', ignore_keys: list=None):
        """
        Uses a key status == 'started' or 'done' to keep track

        :param db_name: Database Name to use
        :param collection_name: Collection name to use
        :param unique_keys: List of keys to treat as a unique value
        :param ignore_keys: Blacklisted keys that will be removed and not inserted
                            into database.
        """
        self._client = pymongo.MongoClient(port=27017)
        self._db = self._client.get_database(db_name)
        self._collect = self._db.get_collection(collection_name)
        self._unique_keys = sorted(list(set(unique_keys)))
        self._ignore_keys = set(ignore_keys) if ignore_keys else set()
        atexit.register(self._client.close)

    def update_with_defaults(self, default_values: dict):
        """
        Given a dict with the default value, set this to all entries in the
        database if it does not exist.

        :param default_values:
        :return:
        """
        updates = []
        for key, value in default_values.items():
            for item in self._collect.find({key: {"$exists": False}}, {'_id': True}):
                updates.append(pymongo.UpdateOne(item, {"$set": {key: value}}))

        if len(updates):
            print("Update:", self._collect.bulk_write(updates).modified_count)

    def cleanup(self):
        """
        Remove stale status runs from database that are started but not done

        :return:
        """
        deletes = []
        for item in self._collect.find({'status': 'started'}, {'_id': True}):
            deletes.append(pymongo.DeleteOne(item))
        # Remove them
        if len(deletes):
            print("Delete", self._collect.bulk_write(deletes).deleted_count)

    def _get_query(self, opt: dict):
        """
        build the query dictionary from unique keys
        :param opt: dictionary, must contain unique keys
        :return:
        """
        query = {}
        for k in self._unique_keys:
            query[k] = opt[k]
        return query

    def _query_by_dict(self, opt: dict):
        """
        Check if this exists in the database from the check_keys list.

        :param opt:
        :return:
        """
        return self._collect.find(self._get_query(opt))

    def _has_run(self, opt: dict, status: Union[List[str], str]):
        """
        Checks if we have at least one completed run

        :param opt:
        :return:
        """
        if isinstance(status, str):
            status = [status]

        for item in self._query_by_dict(opt):
            if item.get('status') in status:
                return True
        return False

    def should_run(self, opt: dict, blacklist_status=['done', 'started']):
        """
        Assumes will start run after calling this

        :param opt:
        :return:
        """
        if self._has_run(opt, blacklist_status):
            return False

        results = copy.deepcopy(opt)
        results['status'] = 'started'
        for k in self._ignore_keys:
            if k in results:
                del results[k]
        self._collect.insert_one(results)
        return True

    def report(self, results: dict):
        """
        Report results and update database

        :param results:
        :return:
        """
        results = copy.deepcopy(results)
        results['status'] = 'done'
        results['time'] = str(dt.datetime.now())

        # insert or replace
        existing = self._query_by_dict(results)
        _id = None
        done_count = 0
        for item in existing:
            if item.get('status') == 'done':
                done_count += 1
            else:
                # Existing run, replace it
                _id = item['_id']

        for k in self._ignore_keys:
            if k in results:
                del results[k]

        # Existing one we overwrite it
        if _id:
            print("Replace: ", self._collect.replace_one({'_id': _id}, results, upsert=True).modified_count)
        else:
            # Else insert
            print("Inserted: ", self._collect.insert_one(results).inserted_id)

        # Check number we have finished
        if done_count:
            print(f"[Warning Found {done_count} existing runs adding anyway]")


class ExperimentTracker(object):

    def __init__(self, filename, unique_keys):
        """
        This class uses the specified keys into a dataframe to check if the
        experiment was already run. It also stores the results in a file and
        performs some level of file locking so this can be called in parallel
        from multiple processes running the same script.

        :param filename: Filename to load or save from
        :param unique_keys: Keys to consider this run as unique
        """
        self.filename = filename
        self.data = None
        self.unique_keys = sorted(list(set(unique_keys)))
        self._reload()

    def _reload(self):
        """Read in csv file"""
        if os.path.exists(self.filename):
            self.data = pd.read_csv(self.filename)
        else:
            self.data = pd.DataFrame(columns=self.unique_keys)

        # Set these default values
        # if 'weight_rescale' not in self.data.columns:
        #     self.data['weight_rescale'] = 'none'
        # if 'norm' not in self.data.columns:
        #     self.data['norm'] = 'softmax'
        # if 'update' not in self.data.columns:
        #     self.data['update'] = 'all'
        # if 'replay' not in self.data.columns:
        #     self.data['replay'] = False
        if 'debug' not in self.data.columns:
            self.data['debug'] = False

        # if 'tie' not in self.data.columns:
        #     self.data['tie'] = False

        if 'update_length' not in self.data.columns:
            self.data['update_length'] = 0
        # for key in self.unique_keys:
        #     self.data[key] = np.nan
        # Remaining set to None
        # for k in self.check_keys:
        #     if k not in self.data.columns:
        #         self.data[k] = None

    def _query_df(self, opt: dict):
        """
        Check if we have results for the given hyperparameters by forming a
        query to the dataframe from the check_keys list.

        :param opt:
        :return:
        """
        query = []
        for k in self.unique_keys:
            val = opt[k]
            if isinstance(val, str):
                query.append(f"{k}=='{val}'")
            else:
                query.append(f"{k}=={val}")
        return self.data.query(" and ".join(query).strip())

    def _get_status(self, opt: dict):
        """
        Check the status

        :param opt:
        :param pretrained_params:
        :return:
        """
        for _, row in self._query_df(opt).iterrows():
            return row.get('status', 'done')
        return 'unknown'

    def should_run(self, opt: dict):
        """
        Assumes will start run after calling this

        :param opt:
        :return:
        """
        self._reload()
        status = self._get_status(opt)
        if status == 'unknown':
            # Set current run to be started
            results = copy.deepcopy(opt)
            results['status'] = 'started'
            self._update(results)
            return True
        elif status == 'done':
            return False

        return True

    def report(self, results: dict):
        """
        Report results

        :param results:
        :return:
        """
        self._reload()
        results = copy.deepcopy(results)
        results['status'] = 'done'
        results['time'] = str(dt.datetime.now())
        self._update(results)

    def _update(self, results: dict):
        """
        Given the new results update the status
        """
        new_row = pd.DataFrame.from_dict([results])
        self._reload()
        # No values
        if not len(self.data):
            self.data = new_row

        # Existing runs, they may have different status eg done, started, unknown
        existing = self._query_df(results)

        # Check number we have finished
        if len(existing[existing['status'] == 'done']):
            print(f"[Warning Found {len(existing)} existing runs adding anyway]")

        # Overwrite the one we started if possible else concat
        started = existing[existing['status'] == 'started']
        if len(started):
            # Delete the first started one...
            self.data = self.data.drop(index=started.index[0])

        self.data = pd.concat([self.data, new_row], sort=False)
        self._write()

    def _write(self):
        """Dump to the file and flush"""
        # Reload
        with portalocker.Lock(self.filename, 'w') as fh:
            self.data.to_csv(fh, index=False)
            fh.flush()
            os.fsync(fh.fileno())
