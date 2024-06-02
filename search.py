from sklearn.feature_extraction.text import CountVectorizer
import math
import numpy as np
from gensim.parsing.preprocessing import preprocess_string

class ResultItem:
    def __init__(self, index, name, text, url, date):
        self.index = index
        self.name = name
        self.head = text.split('\n')[0]
        self.text = text
        self.url = url
        self.date = date
        self.rank = 0.0
        self.freq = 0
        self.count = 0
        self.occurrence = []
        self.similarity = 0.0

    def __str__(self):
        s = f"Title: {self.name}\nURL: {self.url}\nDate: {self.date}\nHead: {self.head}\nFrequency: {self.freq}\nRank: {self.rank}\nSimilarity: {self.similarity}\n"
        for j in self.occurrence:
            s += "> ..." + self.text[max(0, j[0] - 50):j[0] + 50] + "...\n"
        return s

def get_similarity(a, b):
    dot = sum(a[i] * b[i] for i in range(len(a)))
    len_a = math.sqrt(sum(a[i] * a[i] for i in range(len(a))))
    len_b = math.sqrt(sum(b[i] * b[i] for i in range(len(b))))
    return dot / (len_a * len_b)

def get_sustainability_score(text):
    
    sustainability_keywords = ['environment', 'sustainability', 'social responsibility', 'renewable energy', 'carbon footprint']
    
    count = sum(text.lower().count(keyword) for keyword in sustainability_keywords)
    
    score = count / len(sustainability_keywords)
    
    return score

def run_search(search_str, inverse_index, file_names, texts, urls, dates, bag, array,sustainability_topics,lda_model,dictionary, corpus):
    temp = []
    freq = []
    s_list = search_str.split(' ')
    for s in s_list:
        if s in inverse_index:
            temp.append(inverse_index[s].copy())
            freq.append(sum(i[1] for i in inverse_index[s]))
        else:
            temp.append([])
            freq.append(0)

    result_dict = {}
    for index, term_docs in enumerate(temp):
        for doc in term_docs:
            doc_id = doc[0]
            if doc_id not in result_dict:
                item = ResultItem(doc_id, file_names[doc_id], texts[doc_id], urls[doc_id], dates[doc_id])
                item.count += 1
                item.freq += doc[1]
                item.rank += doc[1] * 100 / freq[index]
                item.occurrence.extend(doc[2])
                result_dict[doc_id] = item
            else:
                result_dict[doc_id].count += 1
                result_dict[doc_id].freq += doc[1]
                result_dict[doc_id].rank += doc[1] * 100 / freq[index]
                result_dict[doc_id].occurrence.extend(doc[2])

    # 计算文档的可持续发展得分并调整排名
    for item in result_dict.values():
        # 将文本转换为预处理后的词袋表示
        processed_text = preprocess_string(item.text)
        doc_bow = dictionary.doc2bow(processed_text)
        
        # 获取文档的主题分布
        doc_topics = lda_model.get_document_topics(doc_bow)
        
        # 计算与可持续发展相关的相似度
        sustainability_similarity = sum(prob for topic_id, prob in doc_topics if topic_id in sustainability_topics)
        
        # 调整排名
        item.rank *= 1 + sustainability_similarity

    result_list = list(result_dict.values())
    search_vec = CountVectorizer(vocabulary=bag.get_feature_names_out()).fit_transform([search_str]).toarray()
    for item in result_list:
        item.similarity = get_similarity(search_vec[0], array[item.index].A[0])

    result_list.sort(key=lambda x: -x.similarity)
    return result_list
