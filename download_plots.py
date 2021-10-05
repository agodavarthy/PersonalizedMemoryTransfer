from __future__ import unicode_literals
import pymongo
import pandas as pd
import os
from imdb import IMDb

import datetime as dt
from tqdm import tqdm
import time
import numpy as np
import pickle

from tmdbv3api import TMDb
tmdb = TMDb()
tmdb.api_key = '86b2beda4515f94c053918798943add8'
tmdb.language = 'en'
# tmdb.debug = True
from tmdbv3api import Movie

ia = IMDb()
ia.del_cookies()

client = pymongo.MongoClient(port=27021)
db = client.get_database('movies')
collection = db.get_collection('movies')


def add_by_tmdbid(tmdbIds):
    """
    Search in TMDB for movie summary/information and add it to our database
    """
    for tmdbId in tqdm(tmdbIds, total=len(tmdbIds)):
        count = collection.find_one({
            'tmdb_id': tmdbId,
            'tmdb': {'$exists': True}})

        # if We have it skip
        if count is not None:
            continue

        query = {'tmdb_id': tmdbId}
        # try:
        m = Movie()
        movie = m.details(tmdbId)
        genres = getattr(m, 'genres', [])

        # No title....
        if not hasattr(movie, 'title'):
            continue

        doc = {
            'title': movie.title,
            'originalTitle': movie.original_title,
            'overview': movie.overview,
            'genre': [g['name'] for g in genres],
            #'alt': [[t.title, t.country] for t in movie.alternate_titles],
            'runtime': movie.runtime,
            # 'keyword': [[k.id, k.name] for k in movie.keywords]
        }
        if hasattr(movie, "release_date") and movie.release_date != '':
            date = dt.datetime.strptime(movie.release_date, "%Y-%m-%d")
            doc.update({
                'releaseDate': date,
                'year': date.year
            })
        if len(doc):
            collection.update_one(query, {'$set': {'tmdb': doc}})
        time.sleep(np.random.uniform(5.0, 15))
        # except KeyboardInterrupt:
        #     break
        # except:
        #     print(f"ERROR on {tmdbId}")


def add_by_imdbid(imdbIds):
    for i, imdbId in tqdm(enumerate(imdbIds), total=len(imdbIds)):
        check = collection.find_one({
            'imdb_id': imdbId},
                # 'imdb': {'$exists': True}},
                {"_id": False, "imdb": True})

        # We have it
        if check is not None and 'imdb' in check and len(check['imdb']):
            continue

        query = {'imdb_id': imdbId}

        movie = ia.get_movie(imdbId)

        keys = ['year', 'title', 'genres',
                'canonical title',
                'long imdb title',
                'long imdb canonical title',
                'smart canonical title',
                'smart long imdb canonical title',
                'akas',
                'summary',
                'plot summary', 'synopsis', 'plot', 'plot outline']
        movie_keys = movie.keys()
        get = [key for key in keys if key in movie_keys]
        doc = {k: movie[k] for k in get}
        if len(doc):
            collection.update(query, {'$set': {'imdb': doc}})

        # Every 500 sleep 5-10 minutes
        if i > 0 and i % 500 == 0:
            time.sleep(np.random.uniform(60*5, 60*10))

        time.sleep(np.random.uniform(10.0, 20.5))


if __name__ == '__main__':
    # {'_id': 171495, 'tmdb_id': '409926', 'imdb_id': '0081846'}

    # tmdb_ids = [item['tmdb_id'] for item in collection.find(
    #         {'tmdb': {'$exists': False}, 'tmdb_id': {"$exists": True}},
    #         {"_id": False, 'tmdb_id': True})]
    # add_by_tmdbid(tmdb_ids)
    # imdb_ids = [item['imdb_id'] for item in collection.find(
    #         {'imdb': {'$exists': False}, 'imdb_id': {"$exists": True}},
    #         {"_id": False, 'imdb_id': True})]
    add_by_imdbid(['0092783'])

