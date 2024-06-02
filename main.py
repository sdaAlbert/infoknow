from process import *
from search import *
from datetime import datetime

def collect_user_ratings(search_results):
    user_ratings = {}
    for result in search_results:
        rating = input(f"Please rate the relevance of the following result: {result} (1-5)")
        user_ratings[result] = int(rating)
    return user_ratings

def calculate_precision_recall(user_ratings):
    relevant_items = sum(1 for rating in user_ratings.values() if rating >= 3)  # assuming 3, 4, 5 are relevant
    retrieved_items = len(user_ratings)
    precision = relevant_items / retrieved_items if retrieved_items > 0 else 0

    # Assuming all relevant items are retrieved
    total_relevant_items = relevant_items
    recall = relevant_items / total_relevant_items if total_relevant_items > 0 else 0

    return precision, recall

if __name__ == '__main__':
    print("Loading data...")
    text_list = get_text_list()
    # 文件名列表
    file_names = [f"{i}.txt" for i in range(1, 401)]

    # 对应的URL列表（假设本地文件存储在"data"目录下）
    base_url = "file://" + os.path.abspath("data") + "/"
    urls = [base_url + file_name for file_name in file_names]
    # 对应的日期列表（假设日期与文件的创建日期或修改日期对应）
    dates = []
    for file_name in file_names:
        file_path = os.path.join("data", file_name)
        # 获取文件的创建日期
        creation_time = os.path.getctime(file_path)
        creation_date = datetime.fromtimestamp(creation_time).strftime("%Y-%m-%d")
        dates.append(creation_date)
    print("Vectorizing...")
    bag, count = get_bag(text_list)
    print("Generating index...")
    inverse_index = generate_inverse_index(text_list, bag, count.toarray())

    # for text in text_list:
    #     regex_info = regex_extract(text)
    #     ner_info = ner_extract(text)
    #     print("Regex Extracted Info:", regex_info)
    #     print("NER Extracted Info:", ner_info)


    dictionary, corpus =get_dictionary(text_list)
    
    num_topics = 5  # 假设有5个主题
    lda_model = LdaModel(corpus, id2word=dictionary, num_topics=num_topics)

    
    sustainability_keywords = ['environment', 'sustainability', 'social responsibility', 'renewable energy', 'carbon footprint']
    sustainability_topics = []  # 存储与可持续发展相关的主题

    for keyword in sustainability_keywords:
        keyword_id = dictionary.token2id.get(keyword)
        if keyword_id is not None:
            topic_id, _ = max(lda_model.get_term_topics(keyword_id), key=lambda x: x[1])
            sustainability_topics.append(topic_id)



    print("Done. Type to search now.")
    while True:
        search_str = input("> ")

        if search_str == 'q':
            print('Bye :)')
            exit(0)

        result = run_search(search_str, inverse_index, file_names, text_list, urls, dates, bag, count,sustainability_topics,lda_model,dictionary, corpus)
        for i in result:
            print(i)
            s = input("n to next, q to quit, r to rate(1-5)\n> ")
            if s == 'q':
                break
            elif s == 'n':
                continue
            elif s == 'r':
                user_ratings = collect_user_ratings(result)
                precision, recall = calculate_precision_recall(user_ratings)
                print(f"Precision: {precision}, Recall: {recall}")
                break
