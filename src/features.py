from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.base import TransformerMixin
import re

class TextCleaner(TransformerMixin):
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        # lowercase + remove links, @mentions, hashtags
        cleaned = X.fillna("").astype(str).apply(
            lambda t: re.sub(r"http\S+|@\S+|#", "", t.lower())
        )
        return cleaned

def build_tfidf_pipeline(max_features=20000, ngram_range=(1,2)):
    return Pipeline([
        ("cleaner", TextCleaner()),
        ("tfidf", TfidfVectorizer(max_features=max_features, ngram_range=ngram_range))
    ])
