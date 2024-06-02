import os
import re
from sklearn.feature_extraction.text import CountVectorizer
from collections import defaultdict
from transformers import pipeline

import gensim
from gensim.corpora import Dictionary
from gensim.models import LdaModel
from gensim.parsing.preprocessing import preprocess_string
from gensim.models import CoherenceModel

# 定义数据路径
path = 'data'
files = os.listdir(path)

def get_text_list():
    text_list = []
    for i in files:
        with open(os.path.join(path, i), encoding='utf-8') as f:
            text = f.read()
            text_list.append(text)
    return text_list

def get_bag(texts):
    bag = CountVectorizer(token_pattern='\\b[A-Za-z]+\\b')
    count = bag.fit_transform(texts)
    return bag, count

def get_dictionary(text_list):
    processed_texts = [preprocess_string(text) for text in text_list]
    dictionary = Dictionary(processed_texts)

    corpus = [dictionary.doc2bow(text) for text in processed_texts]
    return dictionary, corpus

def generate_inverse_index(text_list, bag, array):
    result = defaultdict(list)
    words = bag.get_feature_names_out()
    for index, value in enumerate(text_list):
        for i, word in enumerate(words):
            if array[index][i] != 0:
                position_list = [m.span() for m in re.finditer(r'\b' + word + r'\b', value)]
                result[word].append((index, array[index][i], position_list))
    return result

# 正则表达式抽取基础信息
def regex_extract(text):
    patterns = {
        'dates': re.compile(r'\b\d{4}-\d{2}-\d{2}\b|\b\d{2}-\d{2}-\d{4}\b'),
        'scores': re.compile(r'\b\d+-\d+\b'),
        'names': re.compile(r'\b[A-Z][a-z]*\s[A-Z][a-z]*\b'),
        'teams': re.compile(r'\b[A-Z][a-z]*\s(?:United|City|Rovers|Wanderers|Albion|Athletic|Hotspur|Arsenal|Chelsea|Liverpool|Manchester\sUnited)\b'),
        'locations': re.compile(r'\b[A-Z][a-z]+(?:\s[A-Z][a-z]+)*\b')
    }

    extracted_info = {}
    for key, pattern in patterns.items():
        extracted_info[key] = pattern.findall(text)
    return extracted_info

# 使用深度学习的NER模型抽取信息
def ner_extract(text):
    ner_pipeline = pipeline("ner", grouped_entities=True)
    entities = ner_pipeline(text)
    return entities