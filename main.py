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
              db_table.findAll("th", {"class": "pad-l"})] # find all the hypertext references that is a string without recursive then strip them in the <th> with class="pad-l"
    all_db = list(dict.fromkeys(all_db)) # search fromkeys
    db_list = all_db[:10]# get top 10
    db_list = [s.lower() for s in db_list] # lower case all texts
    db_list = [[x.strip() for x in s.split()] for s in db_list] # split into the word then remove the leading and trailing whitespaces
    return db_list


if __name__ == '__main__':
    count_python_mysql()

    print('--------------------')
    cleaned_db = parse_db()
    parse_description = parse_job_description()
    raw = [None] * len(cleaned_db)
    for i, db in enumerate(cleaned_db):
        raw[i] = parse_description.apply(lambda s: np.all([x in s for x in db])).sum()
        print(' '.join(db) + ':' + str(raw[i]) + ' of ' + str(parse_description.shape[0]))

    print('--------------------')
    with_python = [None] * len(cleaned_db)
    for i, db in enumerate(cleaned_db):
        with_python[i] = parse_description.apply(lambda s: np.all([x in s for x in db]) and 'python' in s).sum()
        print(' '.join(db) + ' + python: ' + str(with_python[i]) + ' of ' + str(parse_description.shape[0]))

    print('--------------------')
    for i, db in enumerate(cleaned_db):
        print(' '.join(db) + ' + python: ' + str(with_python[i]) + ' of ' + str(raw[i]) + ' (' +
              str(np.around(with_python[i] / raw[i] * 100, 2)) + '%)')

    lang = [['java'], ['python'], ['c'], ['kotlin'], ['swift'], ['rust'], ['ruby'], ['scala'], ['julia'], ['lua']]
    parsed_description = parse_job_description()
    parsed_db = parse_db()
    all_terms = lang + parsed_db
    query_map = pd.DataFrame(parsed_description.apply(lambda s: [1 if np.all([d in s for d in db])
                                                                 else 0 for db in all_terms]).values.tolist(),
                             columns=[' '.join(d) for d in all_terms])
    print(query_map)

    query_map2 = query_map[query_map['java'] > 0].apply(lambda s: np.where(s == 1)[0],
                                                        axis=1).apply(lambda s: list(query_map.columns[s]))
    print(query_map2)

    str1 = 'the chosen software developer will be part of a larger engineering team developing software for medical ' \
           'devices.'
    str2 = 'we are seeking a seasoned software developer with strong analytical and technical skills to join our ' \
           'public sector technology consulting team.'

    tokened_str1 = word_tokenize(str1)
    tokened_str2 = word_tokenize(str2)

    tokened_str1 = [w for w in tokened_str1 if len(w) > 2]
    tokened_str2 = [w for w in tokened_str2 if len(w) > 2]

    no_sw_str1 = [word for word in tokened_str1 if not word in stopwords.words()]
    no_sw_str2 = [word for word in tokened_str2 if not word in stopwords.words()]

    ps = PorterStemmer()
    stemmed_str1 = np.unique([ps.stem(w) for w in no_sw_str1])
    stemmed_str2 = np.unique([ps.stem(w) for w in no_sw_str2])

    full_list = np.sort(np.concatenate([stemmed_str1, stemmed_str2]))
    print(full_list)
