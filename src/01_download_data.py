import re
import os
from datasets import load_dataset

ds = load_dataset("allenai/WildChat-1M", split="train")

# 1. Загружаем небольшую часть для начала (чтобы не ждать 4 часа)

print(f"Всего диалогов: {len(ds)}")

# 2. Функция для определения "новостного" диалога
def is_news_dialogue(example):
    # Берем только английские диалоги (для простоты)
    if example.get("language") != "English":
        return False
    
    # Склеиваем весь диалог в один текст
    full_text = ""
    for turn in example["conversation"]:
        full_text += turn.get("content", "") + " "
    full_text = full_text.lower()
    
    # Ключевые слова для новостей
    news_keywords = [
        r'\bnews\b', r'\bheadline', r'\bbreaking', r'\breporter', 
        r'\bjournalist', r'\bartikel', r'\bnewspaper', r'\barticle',
        r'\bcurrent events', r'\blatest'
    ]
    
    # Проверяем наличие ключевых слов
    for keyword in news_keywords:
        if re.search(keyword, full_text):
            return True
    return False

# 3. 
sample = ds.select(range(1000))
news_sample = sample.filter(is_news_dialogue)
print(f"Найдено новостных в сэмпле: {len(news_sample)}")

# Если всё ок, фильтруем больше
if len(news_sample) > 0:
    # Берем первые 50000 и фильтруем (чтобы не зависнуть)
    ds_subset = ds.select(range(50000))
    news_dialogues = ds_subset.filter(is_news_dialogue)
    
    # Берем 2000 примеров
    final_news = news_dialogues.select(range(min(2000, len(news_dialogues))))
    
    # Сохраняем
    data_path = os.path.join("..", "data", "raw", "wildchat_news")
    abs_path = os.path.abspath(data_path)
    final_news.save_to_disk(abs_path)
    print(f"Сохранено {len(final_news)} диалогов")
