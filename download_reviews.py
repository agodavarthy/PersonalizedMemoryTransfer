"""
db.reviews.aggregate([
    {"$group": {
        "_id": {"user_id": "$author"},
        "count": {"$sum": 1}}
    }, 
    {"$sort": {"count": -1}},
    {"$group": 
        {"_id": {"user_id":"$_id.user_id"}, "count": {"$sum": 1}}},
        {"$project": {"total": {"$sum": "$count"}}}
]
)
"""
import pymongo
from imdb import IMDb

import datetime as dt
from tqdm import tqdm
import time
import numpy as np
import re

from imdb.parser.http.utils import DOMParserBase, analyze_imdbid
from imdb.parser.http.piculet import Path, Rule, Rules


class DOMHTMLReviewsParser(DOMParserBase):
    """Parser for the "reviews" pages of a given movie.
    The page should be provided as a string, as taken from
    the www.imdb.com server.  The final result will be a
    dictionary, with a key for every relevant section.

    Example::

        rparser = DOMHTMLReviewsParser()
        result = rparser.parse(reviews_html_string)
    """
    rules = [
        Rule(
                key='reviews',
                extractor=Rules(
                        foreach='//div[@class="review-container"]',
                        rules=[
                            Rule(
                                    key='text',
                                    # extractor=Path('.//div[contains(@class, "text")]//text()')
                                    extractor=Path('.//div[@class="text show-more__control"]//text()')
                            ),

                            Rule(
                                    key='helpful',
                                    extractor=Path('.//div[@class="actions text-muted"]/text()[1]')
                            ),
                            Rule(
                                    key='title',
                                    extractor=Path('.//a[contains(@class, "title")]//text()')
                            ),
                            Rule(
                                    key='author',
                                    extractor=Path('.//span[@class="display-name-link"]/a/@href')
                            ),
                            Rule(
                                    key='date',
                                    extractor=Path('.//span[@class="review-date"]//text()')
                            ),
                            Rule(
                                    key='rating',
                                    extractor=Path(
                                            './/span[@class="point-scale"]/preceding-sibling::span/text()'
                                            # './/span[@class="rating-other-user-rating"]/preceding::span/text()'
                                            # './/span[@class="point-scale"]/preceding-sibling::span[1]//text()'
                                    )
                            ),
                            Rule(key='permalink',
                                 extractor=Path('.//a[contains(text(), "Permalink")]/@href'))
                        ],
                        transform=lambda x: ({
                            'content': x.get('text', '').replace('\n', ' ').replace('  ', ' ').strip(),
                            'helpful': [int(s) for s in x.get('helpful', '').split() if s.isdigit()],
                            'title': x.get('title', '').strip(),
                            'author': analyze_imdbid(x.get('author')),
                            'date': x.get('date', '').strip(),
                            'rating': x.get('rating', '').strip(),
                            'review_id': x.get('permalink', '').strip()
                        })
                )
        )
    ]

    preprocessors = [('<br>', '<br>\n')]

    def postprocess_data(self, data):
        for review in data.get('reviews', []):
            if review.get('rating') and len(review['rating']):
                review['rating'] = int(review['rating'])
            else:
                review['rating'] = None

            if review.get('helpful') and len(review['helpful']) == 2:
                review['not_helpful'] = review['helpful'][1] - review['helpful'][0]
                review['helpful'] = review['helpful'][0]
            else:
                review['helpful'] = 0
                review['not_helpful'] = 0

            review['author'] = "ur%s" % review['author']
            if review.get('review_id', ''):
                match = re.search(r"rw\d+", review.get('review_id', ''))
                if match:
                    review['review_id'] = match.group()
                else:
                    review['review_id'] = None
            else:
                review['review_id'] = None
        return data


# cont = self._retrieve(self.urls['movie_main'] % movieID + 'reviews?count=9999999&start=0')
#         return self.mProxy.reviews_parser.parse(cont)


if __name__ == '__main__':
    client = pymongo.MongoClient(port=27021)
    db = client.get_database('movies')
    col_movies = db.get_collection('movies')
    col_reviews = db.get_collection('reviews')
    ia = IMDb()
    parser = DOMHTMLReviewsParser()
    ia.del_cookies()

    total = 0
    skip = 25+493+40+96+5+902+1700+3522+2+3400+30055+2+6302+5927+3329+100
    duplicate_count = 0
    review_wait_count = np.random.randint(300, 600)
    try:
        cursor = col_movies.find({'imdb_id': {'$exists': True}}, {'_id': 1, 'imdb_id': 1},
                                 no_cursor_timeout=True, skip=skip)
        url = 'reviews?sort=submissionDate&dir=desc&ratingFilter=0'
        progress = tqdm(enumerate(cursor))
        for i, ex in progress:
            movieID = ex['imdb_id']
            # For now we are skipping if we already crawled it once.
            if col_reviews.find_one({'imdb_id': movieID}):
                continue
            progress.set_description(f"Reviews: {total:,} - Duplicates: {duplicate_count:,} - Status: Crawl -> {movieID}  ")
            docs = []
            # Check for
            try:
                cont = ia._retrieve(ia.urls['movie_main'] % movieID + url)
            except Exception as e:
                # Only if we cannot find it...
                if str(e).find("HTTPError 404: 'Not Found'") != -1:
                    print(f"\n\nBad MovieID {movieID}\n\n")
                    continue
                elif str(e).find("Connection reset by peer") != -1:
                    print("\n\nConnection Reset... skipping")
                    continue
                elif str(e).find("read operation timed") != -1:
                    print("Read operation timeout")
                    continue
                else:
                    print(str(dt.datetime.now()))
                    # Raise it
                    raise e

            parsed = parser.parse(cont)['data'].get('reviews', [])
            for review in parsed:
                # Skip if we already have it
                if col_reviews.count({"_id": review['review_id']}) == 1:
                    duplicate_count += 1
                    continue
                review['imdb_id'] = movieID
                review['_id'] = review['review_id']
                review['crawl_date'] = str(dt.datetime.now())
                del review['review_id']
                docs.append(review)

            if len(docs):
                total += len(docs)
                col_reviews.insert_many(docs, ordered=False)

            sleep_time = np.random.uniform(15.0, 60)
            # Additional sleepish..
            if i > 0 and i % 25 == 0:
                sleep_time += np.random.uniform(60, 180)
            if i > 0 and i % review_wait_count == 0:
                print(f"\nCrawled {review_wait_count} values sleeping for an hourish")
                review_wait_count = np.random.randint(300, 600)
                sleep_time += np.random.uniform(60*60, 60*60*2)

            progress.set_description(f"Reviews: {total:,} - Duplicates: {duplicate_count:,} - Status: Sleep Time: {sleep_time:.1f} -> {movieID} ")
            time.sleep(sleep_time)
    finally:
        cursor.close()
    # ia.mProxy.reviews_parser.parse(cont)


# parser = DOMHTMLReviewsParser()

# movieID = "0114576"
# cont = ia._retrieve(ia.urls['movie_main'] % movieID + 'reviews?')
# parser.parse(cont)['data']['reviews'][1]['content'][-100:]