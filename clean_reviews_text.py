import sys
assert sys.version_info >= (3, 5) # make sure we have Python 3.5+

from pyspark.sql import SparkSession, functions, types
from pyspark.sql.functions import broadcast
from pyspark.sql.functions import udf
from langdetect import detect
import string
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize




def remove_punct(text):
    regex = re.compile('[' + re.escape(string.punctuation) + '0-9\\r\\t\\n]')
    nopunct = regex.sub(" ", text)
    return nopunct

def convert_rating(rating):
    if rating >=4:
        return 1
    else:
        return 0

nltk.download('punkt')
nltk.download('stopwords')
# nltk.download('corpus')

stop_words = set(stopwords.words('english'))
def remove_stopping_words(text, stopwords=stop_words):
    word_tokens = word_tokenize(text)
    filtered_sentence = [w for w in word_tokens if not w.lower() in stop_words]
    new_sentence = ' '.join(filtered_sentence)
    # print(word_tokens)
    # print(filtered_sentence)
    return new_sentence

get_language = udf(lambda x : detect(x))
# def get_language(x):
#     try:
#         detect(x)
#     except:
#         return '--'

# lang_identifier = udf(lambda x: get_language(x))
punct_remover = udf(lambda x: remove_punct(x))
stars_conveter = udf(lambda x : convert_rating(x))
stop_words_remover = udf(lambda x : remove_stopping_words(x))

def main(reviews_file):
    reviews_schema = types.StructType([
    types.StructField('review_id', types.StringType()),
    types.StructField('user_id', types.StringType()),
    types.StructField('business_id', types.StringType()),
    types.StructField('stars', types.FloatType()),
    types.StructField('useful', types.IntegerType()),
    types.StructField('funny', types.IntegerType()),
    types.StructField('cool', types.IntegerType()),
    types.StructField('contents', types.StringType()),
    types.StructField('date', types.TimestampType()),
    ])

    # PREPROCESSING REVIEWS
    reviews_df = spark.read.parquet(reviews_file).select('review_id','contents', 'stars')
    reviews_df.show(1)
    # reviews_df.show(5)

    # 1) FILTER OUT NON ENGLISH REVIEWS
    # Check if Reviews are only in English.
    # Find non-english reviews

    reviews_df = reviews_df.withColumn('language', get_language(reviews_df['contents']))
    # print(reviews_df.show())
    # all_languages = reviews_df.groupBy('language').count()
    # print(all_languages.show(5))
    reviews_df = reviews_df.filter(reviews_df['language'] == 'en')

    # 2) REMOVE PUNCTUATION MARKS & OTHER SYMBOLS
    # no_punct_df = reviews_df.select('review_id', punct_remover(reviews_df['contents']).alias('contents'), 'stars')
    # # print(no_punct_df.collect()[0][1])



    # 3) CONVERT REVIEW RATING INTO EITHER POSITIVE OR NEGATIVE
    # new_ratings = no_punct_df.select('review_id', 'contents', stars_conveter(reviews_df['stars']).alias('sentiment'))
    # # print(new_ratings.collect()[0][2])
    # new_ratings.show()


    #4) REMOVE STOP WORDS
    no_stop_words = reviews_df.withColumn('clean_content', stop_words_remover(reviews_df['contents']))
    print(no_stop_words.collect()[0][1])
    print(no_stop_words.collect()[0][4])








    #5) Convert to lowecase



    #5) CREATE N-GRAMS OF WORDS


    # SEE HOW AFFIN LEXICON WORKS
    # categories = ['service', 'food', 'price']
    # count_dict = {}
    from afinn import Afinn
    af = Afinn()
    sentiment_scores = [af.score(no_stop_words.collect()[0][1])]
    sentiment_category = ['positive' if score > 0
                          else 'negative' if score < 0
                              else 'neutral'
                                  for score in sentiment_scores]

    print(sentiment_scores)
    print(sentiment_category)
    sentence = 'The price was very affordable too, $11 for 3 tacos.'
    sentiment_scores = [af.score(sentence)]
    sentiment_category = ['positive' if score > 0
                          else 'negative' if score < 0
                              else 'neutral'
                                  for score in sentiment_scores]

    print(sentiment_scores)
    print(sentiment_category)


    # CATEGORIES
    categories = {'food': {'words': ['food', 'taste'], 'pos_count': 0, 'neg_count': 0, 'neu_count': 0}],
                  'service': {'words': ['service, waitime, waiter'], 'pos_count': 0, 'neg_count': 0, 'neu_count': 0},
                  'price': {'words': ['price'], 'pos_count': 0, 'neg_count': 0, 'neu_count': 0}
                    }

    def get_counts(text, categories=categories):
        sentences = text.split('.')
        for sentence in sentences:
            category = get_sentiment_category(sentence)
            for k, v in categories:
                for w in v['words']:
                    if w in sentence:
                        if category == 'positive':
                            v['p_count'] += 1
                        elif category == 'neutral':
                            v['neu_count'] += 1
                        else:
                            v['neg_count'] += 1
                        break
        return categories




    #6) Save Files to parquet
    # df.write.mode("overwrite").parquet(output + '/processed_reviews.parquet')
if __name__ == '__main__':
    reviews_file = sys.argv[1]
    spark = SparkSession.builder.appName('Reviews Parquet').getOrCreate()
    assert spark.version >= '3.0' # make sure we have Spark 3.0+
    spark.sparkContext.setLogLevel('WARN')
    sc = spark.sparkContext
    main(reviews_file)
