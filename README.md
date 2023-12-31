# Brand-Analytics-ML-contest
**Brand Analytics ML-contest Суммаризация комментариев в социальных медиа**

**Описание задачи:**  

Необходимо реализовать решение которое сможет генерировать (генеративная суммаризация) текст суммаризации (главного смысла, смысла обсуждения) комментариев под каждым постом в нескольких режимах (типах) описанных далее. Проект TextSummarization представляет собой инструмент для суммаризации текстовых данных. Он использует передовые методы машинного обучения для анализа и синтеза текстов, предоставляя удобный интерфейс для работы с большими объемами текстовой информации.

**Входные данные:**

Файл в формате .jsonl (1 json-объект на 1 строку) с постами (вт.ч. видео с YouTube) и комментариями из VK, Telegram и YouTube.
Данные в файле представлены в хаотичном порядке, участникам в первую очередь необходимо связать комментарии и посты по внешним идентификаторам, которые указаны в качестве отдельного поля каждого объекта исходного файла, а также провести базовые операции (очистка и т.п.) предобработок.

**Требования к типам суммаризации:**  

1.	all_comments: суммаризация всех комментариев под каждым постом, без анализа самого поста;
2.	post_comments: суммаризация только тех комментариев, которые имеют явное отношение к тексту каждого поста;
3.	topic_comments: суммаризация комментариев которые имеют косвенное отношение к посту (пример: пост про технологию компании, а комментарий про обсуждение самой компании)

## Решение 
  
Проект включает в себя:  
- Анализ и обработку текстов с использованием NLP (Natural Language Processing).
- Применение модели `d0rj/rut5-base-summ` для суммаризации текстов.
- Использование моделей из `sentence_transformers` для семантического анализа текста.
- Реализована загрузка файлов и выбор суммаризации пользователем

## Установка и запуск  
Чтобы запустить суммаризацию комментариев необходимо: <br>
!Перед запуском установите зависимости <br>
1. Откройте solution.py и summycom.py в терминале <br>
2. Запустите solution.py <br>
2. Введите путь до необходимо dataset с комментариями , после чего выберите необходимый режим обработки комментариев <br>
NOTE: Если вам не выбрасывается tkinter окно, нажмите Alt+Tab и выберите окно под названием "Открытие" <br>
3. Результат работы программы будет сохранен в 'result.jsonl' <br>
NOTE: Для того, чтобы выбрать другой режим суммаризации комментариев ёщё раз запустите solution.py в терминале
