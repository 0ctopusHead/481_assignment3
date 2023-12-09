import re
import timeit
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
import matplotlib.pyplot as plt
import m1


def walkthrough_1():
    cleaned_description = m1.get_and_clean_data()[:1000]

    cleaned_description = cleaned_description.apply(lambda s: re.sub(r'[^A-Za-z]', ' ', s))
    cleaned_description = cleaned_description.apply(lambda s: re.sub(r'\s+', ' ', s))

    tokenized_description = cleaned_description.apply(lambda s: word_tokenize(s))

    stop_dict = set(stopwords.words())
    sw_removed_description = tokenized_description.apply(lambda s: set(s) - stop_dict)
    sw_removed_description = sw_removed_description.apply(lambda s: [word for word in s if len(word) > 2])

    concated = np.unique(np.concatenate([s for s in tokenized_description.values]))
    stem_cache = {}
    ps = PorterStemmer()
    for s in concated:
        stem_cache[s] = ps.stem(s)

    stemmed_description = sw_removed_description.apply(lambda s: [stem_cache[w] for w in s])
    print(stemmed_description)
    return stemmed_description


def walkthrough_2():
    from sklearn.feature_extraction.text import CountVectorizer
    stemmed_description = walkthrough_1()
    cv = CountVectorizer(analyzer=lambda x: x)
    X = cv.fit_transform(stemmed_description)
    print(pd.DataFrame(X.toarray(), columns=cv.get_feature_names_out()))
    print(X.tocsr()[0, :])
    XX = X.toarray()

    print(np.shape(np.matmul(X.toarray(), X.toarray().T)))
    timeit.timeit(lambda: np.matmul(XX, XX.T), number=1)

    print(np.shape(X*X.T))
    timeit.timeit(lambda: X*X.T, number=1)



if __name__ == '__main__':
    walkthrough_1()
    walkthrough_2()
    pd.DataFrame.from_dict
