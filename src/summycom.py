import pandas as pd
import numpy as np
import re
import json
from bs4 import BeautifulSoup
from bert_score import score
from sentence_transformers import SentenceTransformer, util
import torch
from datasets import load_dataset
from tqdm import tqdm
import evaluate
from transformers import pipeline

import tkinter as tk
from tkinter import filedialog

def load_model():
    model_name = 'd0rj/rut5-base-summ'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    summarizer = pipeline('summarization', model=model_name, device=0 if torch.cuda.is_available() else -1)
    return summarizer, device

def get_file_path():
    root = tk.Tk()
    root.withdraw()  #скрываем окно Tkinter
    file_path = filedialog.askopenfilename()  #диалог выбора файла
    return file_path

MODEL, DEVICE = load_model()
MODEL_SEMANTIC_SIMILARITY = SentenceTransformer('all-MiniLM-L6-v2').to(DEVICE)
MODEL_TOPIC = SentenceTransformer('all-MiniLM-L6-v2').to(DEVICE)

class Summy:

    def __init__(self):
        self.posts = None
        self.comments = None
        self.model,self.device = MODEL, DEVICE
        self.model_topic = MODEL_TOPIC
        self.model_semantic_similarity = MODEL_SEMANTIC_SIMILARITY

    def load_data(self, path):
        # Загрузка данных из файла
        with open(path, 'r', encoding='utf-8') as file:
            data = [json.loads(line) for line in file]
            df = pd.DataFrame(data)

        # Разделение на посты и комментарии и предобработка текстов
        self.posts = df[df['root_id'].isna()]
        self.comments = df[df['root_id'].notna()]
        self.posts['text'] = self.posts['text'].apply(self.__clean_text)
        self.comments['text'] = self.comments['text'].apply(self.__clean_text)

    def __clean_text(self, text):

        # Удаление HTML-тегов
        text = BeautifulSoup(text, "html.parser").get_text()
        # Удаление ссылок
        text = re.sub(r'http\S+', '', text)
        # Удаление эмодзи
        emoji_pattern = re.compile("["
                                u"\U0001F600-\U0001F64F"  # emoticons
                                u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                                u"\U0001F680-\U0001F6FF"  # transport & map symbols
                                u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                                u"\U00002702-\U000027B0"
                                u"\U000024C2-\U0001F251"
                                "]+", flags=re.UNICODE)
        text = emoji_pattern.sub(r'', text)
        # Удаление переносов строк и лишних пробелов
        text = ' '.join(text.split())
        # Удаление серии символов "—"
        text = re.sub(r'—+', '', text)
        # Удаление серии символов "=" и стрелок "►"
        text = re.sub(r'=+', ' ', text)  # Заменяет серии "=" на пробел
        text = re.sub(r'►', ' ', text)  # Заменяет стрелки "►" на пробел
        return text

    # агрегация комментариев в зависимости от типа
    def __aggregate_comments(self, post_id, summary_type):
        # Определение типа фильтрации
        if summary_type == 'all_comments':
            relevant_comments = self.comments[self.comments['root_id'] == post_id]
        elif summary_type == 'post_comments':
            relevant_comments = self.comments[(self.comments['root_id'] == post_id) & (self.comments['relevant_to_post'] == True)]
        elif summary_type == 'topic_comments':
            relevant_comments = self.comments[(self.comments['root_id'] == post_id) & (self.comments['relevant_to_topic'] == True)]
        else:
            raise ValueError("Invalid summary type. Choose 'all_comments', 'post_comments', or 'topic_comments'.")

        # Конкатенация текстов комментариев
        aggregated_text = ' '.join(relevant_comments['text'])
        return aggregated_text

    def __calc_metrics(self, candidates, references):
        # Вычисление метрики ROUGE
        rouge = evaluate.load('rouge')
        rouge_results = rouge.compute(predictions=candidates, references=references)

        # Форматирование результатов ROUGE
        formatted_rouge_results = {}
        for key in rouge_results.keys():
            if "mid" in dir(rouge_results[key]):
                formatted_rouge_results[key] = {
                    "precision": round(rouge_results[key].mid.precision * 100, 2),
                    "recall": round(rouge_results[key].mid.recall * 100, 2),
                    "fmeasure": round(rouge_results[key].mid.fmeasure * 100, 2)
                }
            else:
                formatted_rouge_results[key] = round(rouge_results[key] * 100, 2)

        # Вычисление метрики BERTScore
        P, R, F1 = score(candidates, references, lang='ru')
        bert_scores = {
            "precision": np.mean(P.numpy()),
            "recall": np.mean(R.numpy()),
            "f1": np.mean(F1.numpy())
        }

        return formatted_rouge_results, bert_scores

    def __filter_post_comments_similarity(self, post_text, comments, threshold=0.5):
        post_embedding = self.model_semantic_similarity.encode(post_text, convert_to_tensor=True, batch_size=64)

        comment_embeddings = self.model_semantic_similarity.encode(comments, convert_to_tensor=True, batch_size=64)

        similarities = util.pytorch_cos_sim(post_embedding, comment_embeddings)[0]

        similarities = similarities.cpu().numpy().flatten()

        filtered_comments = [comment for comment, similarity in zip(comments, similarities) if similarity > threshold]
        return filtered_comments

    def __filter_topic_comments_similarity(self, post_text, comments, lower_threshold=0.1, upper_threshold=0.5):
        post_embedding = self.model_topic.encode(post_text, convert_to_tensor=True, batch_size=64)

        comment_embeddings = self.model_topic.encode(comments, convert_to_tensor=True, batch_size=64)

        similarities = util.pytorch_cos_sim(post_embedding, comment_embeddings)[0]
        similarities = similarities.cpu().numpy().flatten()

        filtered_comments = [comment for comment, similarity in zip(comments, similarities) if lower_threshold <= similarity <= upper_threshold]
        return filtered_comments

    def __summarize_comment(self, comments):
        return MODEL(comments, max_length=200, min_length=30, do_sample=False)[0]['summary_text']

    def process_summarization(self, summary_type):

        if summary_type not in ['all_comments', 'post_comments', 'topic_comments']:
            raise ValueError("Недопустимый тип суммаризации. Выберите из all_comments, post_comments, topic_comments.")

        if summary_type == 'all_comments':
            # Получение агрегированных текстов для каждого поста
            all_comments = self.posts['id'].apply(lambda post_id: self.__aggregate_comments(post_id, 'all_comments'))

            # Суммаризация агрегированных текстов
            summarized_texts = self.__summarize_comment(all_comments.tolist())

            # Убедитесь, что длина summarized_texts соответствует длине self.posts
            if len(summarized_texts) != len(self.posts):
                raise ValueError("Длина суммаризированных текстов не соответствует длине постов")

            # Присваивание суммаризированных текстов обратно в DataFrame
            self.posts['sum_all_comments'] = summarized_texts
            # Проверка метрик для всех комментариев
            candidates = self.posts['sum_all_comments'].tolist()
            references = self.posts['text_all_comments'].tolist()
            rouge_scores, bert_scores = self.__calc_metrics(candidates, references)
            print("ROUGE Scores:", rouge_scores)
            print("BERTScores:", bert_scores)


        elif summary_type == 'post_comments':

            self.comments['relevant_to_post'] = False

            for index, row in tqdm(self.posts.iterrows(), desc="Processing self.posts", total=self.posts.shape[0]):

                post_id = row['id']
                post_text = row['text']

                related_comments = self.comments[self.comments['root_id'] == post_id]['text']
                filtered_comments = self.__filter_post_comments_similarity(post_text, related_comments.tolist())

                for comment in filtered_comments:
                    self.comments.loc[(self.comments['root_id'] == post_id) & (
                                self.comments['text'] == comment), 'relevant_to_post'] = True

                aggregated_comments = self.__aggregate_comments(post_id, 'post_comments')
                summarized_comments = self.__summarize_comment(aggregated_comments)
                self.posts.loc[index, 'text_post_comments'] = aggregated_comments
                self.posts.loc[index, 'sum_post_comments'] = summarized_comments

            #candidates_post = self.posts['sum_post_comments'].tolist()
            #references_post = self.posts['text_post_comments'].tolist()

            #rouge_scores_post, bert_scores_post = self.__calc_metrics(candidates_post, references_post)
            #print("ROUGE Scores:", rouge_scores_post)
            #print("BERTScores:", bert_scores_post)


        elif summary_type == 'topic_comments':

            self.comments['relevant_to_topic'] = False

            for index, row in tqdm(self.posts.iterrows(), desc="Processing self.posts", total=self.posts.shape[0]):

                post_id = row['id']
                post_text = row['text']

                related_comments = self.comments[self.comments['root_id'] == post_id]['text']
                filtered_comments = self.__filter_topic_comments_similarity(post_text, related_comments.tolist())

                for comment in filtered_comments:
                    self.comments.loc[(self.comments['root_id'] == post_id) & (
                                self.comments['text'] == comment), 'relevant_to_topic'] = True

                aggregated_comments = self.__aggregate_comments(post_id, 'topic_comments')
                summarized_comments = self.__summarize_comment(aggregated_comments)
                self.posts.loc[index, 'text_topic_comments'] = aggregated_comments
                self.posts.loc[index, 'sum_topic_comments'] = summarized_comments

            candidates_topic = self.posts['sum_topic_comments'].tolist()
            references_topic = self.posts['text_topic_comments'].tolist()

            rouge_scores_topic, bert_scores_topic = self.__calc_metrics(candidates_topic, references_topic)
            print("ROUGE Scores:", rouge_scores_topic)
            print("BERTScores:", bert_scores_topic)

        else:
            print("Недопустимый тип суммаризации")

        return self.posts

    def save_to_jsonl(self, summary_type, filename):
        # Сопоставление типов суммаризации с названиями столбцов
        summary_column = {
            'all_comments': 'sum_all_comments',
            'post_comments': 'sum_post_comments',
            'topic_comments': 'sum_topic_comments'
        }

        # Проверка корректности типа суммаризации
        if summary_type not in summary_column:
            raise ValueError(f"Недопустимый тип суммаризации: {summary_type}")

        # Выбор столбца для суммаризации
        summary_column_name = summary_column[summary_type]

        with open(filename, "w") as file:
            for _, post_row in self.posts.iterrows():
                # Получение хешей комментариев, соответствующих данному посту
                comments_hashes = self.comments[self.comments['root_id'] == post_row['id']]['hash'].tolist()

                # Формирование данных для JSON
                data = {
                    "summary": post_row[summary_column_name],  # текст суммаризации в зависимости от типа
                    "post_hash": post_row["hash"],  # hash исходного поста
                    "comments_hash": comments_hashes  # хеши комментариев
                }
                json.dump(data, file)
                file.write("\n")

if __name__ == '__main__':

    summy = Summy()
    path = get_file_path()  # получаем путь к файлу
    if path:
        summy.load_data(path)
    else:
        print("Файл не выбран.")

    summy.load_data(path)
    # Запрос выбора типа суммаризации у пользователя
    print("Выберите тип суммаризации: all_comments, post_comments, topic_comments")
    summary_type = input("Введите тип суммаризации: ").strip()

    # Проверка корректности введенного типа суммаризации
    valid_types = ['all_comments', 'post_comments', 'topic_comments']
    if summary_type not in valid_types:
        raise ValueError("Недопустимый тип суммаризации. Выберите из all_comments, post_comments, topic_comments.")

    df = summy.process_summarization(summary_type)
    print(df.head())

    # Сохранение результатов в .jsonl файл
    summy.save_to_jsonl(summary_type, 'result.jsonl')