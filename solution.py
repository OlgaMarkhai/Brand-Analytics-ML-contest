from src.summycom import Summy, get_file_path

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