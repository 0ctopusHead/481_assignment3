from bs4 import BeautifulSoup
import pandas as pd
import string
import requests
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

nltk.download('stopwords')
nltk.download('punkt')


# to remove unwanted things in the data
def get_and_clean_data():
    data = pd.read_csv('data/software_developer_united_states_1971_20191023_1.csv')
    description = data['job_description']
    cleaned_description = description.apply(lambda s: s.translate(str.maketrans('', '',
                                                                                string.punctuation + u'\xa0')))  # remove punctuation
    cleaned_description = cleaned_description.apply(lambda s: s.lower())  # make it to lower case
    cleaned_description = cleaned_description.apply(
        lambda s: s.translate(
            str.maketrans(string.whitespace, ' ' * len(string.whitespace), '')))  # remove the whitespace
    cleaned_description = cleaned_description.drop_duplicates()  # remove the duplication terms after process
    return cleaned_description


# to separate the data into the smallest unit possible
def simple_tokenize(data):
    clean_description = data.apply(lambda s: [x.strip() for x in s.split()])
    return clean_description


def parse_job_description():
    clean_description = get_and_clean_data()
    clean_description = simple_tokenize(clean_description)
    return clean_description


# extract the terms that we interested in
def count_python_mysql():
    parse_description = parse_job_description()
    count_python = parse_description.apply(lambda s: 'python' in s).sum()  # How many 'python' in document
    count_mysql = parse_description.apply(lambda s: 'mysql' in s).sum()
    print('python: ' + str(count_python) + ' of ' + str(
        parse_description.shape[0]))  # from all the remaining document(shape[0]) count the 'python' remained
    print('mysql: ' + str(count_mysql) + ' of ' + str(parse_description.shape[0]))


# visit the site and try to extract information or text
def parse_db():
    html_doc = requests.get("https://db-engines.com/en/ranking").content
    soup = BeautifulSoup(html_doc, 'html.parser')
    db_table = soup.find("table", {"class": "dbi"})  # find the table that class="dbi"
    all_db = [''.join(s.find('a').findAll(string=True, recursive=False)).strip() for s in
              db_table.findAll("th", {
                  "class": "pad-l"})]  # find all the hypertext references that is a string without recursive then strip them in the <th> with class="pad-l"
    all_db = list(dict.fromkeys(all_db))  # search fromkeys
    db_list = all_db[:10]  # get top 10
    db_list = [s.lower() for s in db_list]  # lower case all texts
    db_list = [[x.strip() for x in s.split()] for s in
               db_list]  # split into the word then remove the leading and trailing whitespaces
    return db_list


def inverse_indexing(parsed_description): # 7:16 Voice3 Week2
    sw_set = set(stopwords.words()) - {'c'}
    no_sw_description = parsed_description.apply(lambda x: [w for w in x if w not in sw_set])
    ps = PorterStemmer()
    stemmed_description = no_sw_description.apply(lambda x: set([ps.stem(w) for w in x]))
    all_unique_term = list(set.union(*stemmed_description.to_list()))

    invert_idx = {}
    for s in all_unique_term:
        invert_idx[s] = set(stemmed_description.loc[stemmed_description.apply(lambda x: s in x)].index)

    return invert_idx


def search(invert_idx, query):
    ps = PorterStemmer()
    processed_query = [s.lower() for s in query.split()]
    stemmed = [ps.stem(s) for s in processed_query]
    matched = list(set.intersection(*[invert_idx[s] for s in stemmed]))
    return matched


