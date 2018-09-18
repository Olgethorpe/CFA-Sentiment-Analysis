from enum import Enum
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

# Stores the different vectorizers
class Vectorizers(Enum):
    Tf_idf = TfidfVectorizer()
    Unigram = CountVectorizer()
    Bigram = CountVectorizer(ngram_range = (2,2))