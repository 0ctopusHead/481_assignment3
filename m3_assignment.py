import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
from ordered_set import OrderedSet
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import m1
from BM25 import BM25
import rank_bm25
import argparse


def create_stem_cache(cleaned_description):
    tokenized_description = cleaned_description.apply(lambda s: word_tokenize(s))
    concated = np.unique(np.concatenate([s for s in tokenized_description.values]))
    stem_cache = {}
    ps = PorterStemmer()
    for s in concated:
        stem_cache[s] = ps.stem(s)
    return stem_cache


def create_custom_preprocessor(stop_dict, stem_cache):
    def custom_preprocessor(s):
        ps = PorterStemmer()
        s = re.sub(r'[^A-Za-z]', ' ', s)
        s = re.sub(r'\s+', ' ', s)
        s = word_tokenize(s)
        s = list(OrderedSet(s) - stop_dict)
        s = [word for word in s if len(word)>2]
        s = [stem_cache[w] if w in stem_cache else ps.stem(w) for w in s]
        s = ' '.join(s)
        return s
    return custom_preprocessor


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("query", type=str)
    args = parser.parse_args()

    cleaned_description = m1.get_and_clean_data()
    stem_cache = create_stem_cache(cleaned_description)
    stop_dict = set(stopwords.words('English'))

    my_custom_preprocessor = create_custom_preprocessor(stop_dict, stem_cache)
    tf_idf_vectorizer = TfidfVectorizer(preprocessor=my_custom_preprocessor, use_idf=True, ngram_range=(2, 2))
    tf_idf_vectorizer.fit(cleaned_description)
    transformed_data = tf_idf_vectorizer.transform(cleaned_description)
    X_tfidf_df = pd.DataFrame(transformed_data.toarray(), columns=tf_idf_vectorizer.get_feature_names_out())
    max_term = X_tfidf_df.sum().sort_values()[-10:].sort_index().index

    query = [args.query]
    print("Searching for terms: " + str(query))

    transformed_query = tf_idf_vectorizer.transform(query)
    transformed_query_df = pd.DataFrame(transformed_query.toarray(), columns=tf_idf_vectorizer.get_feature_names_out()).loc[0]
    q_dot_d = X_tfidf_df.dot(transformed_query_df.T)
    print("TF-IDF SCORING")
    print(cleaned_description.iloc[np.argsort(q_dot_d)[::-1][:5].values])

    bm25 = BM25(tf_idf_vectorizer)
    bm25.fit(cleaned_description)
    score = bm25.transform(args.query)
    rank = np.argsort(score)[::-1]
    print("BM25 SCORING")
    print(cleaned_description.iloc[rank[:5]])

    bm25_2 = rank_bm25.BM25Plus(cleaned_description, my_custom_preprocessor)
    doc_scores = bm25_2.get_scores(args.query)
    bm25_2_rank = np.argsort(doc_scores)[::-1]
    print(cleaned_description.iloc[bm25_2_rank[:5]])
    print(doc_scores)