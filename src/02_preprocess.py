from datasets import load_from_disk
import os
import json

data_path = os.path.join("..", "data", "raw", "wildchat_news")
abs_path = os.path.abspath(data_path)
ds_news = load_from_disk(abs_path)

# Извлекаем текст диалогов (в формате для дальнейшей обработки)
conversations = []
for item in ds_news:
    # Склеиваем диалог в читаемый формат
    dialogue = []
    for turn in item["conversation"]:
        role = turn["role"]
        content = turn["content"]
        dialogue.append(f"{role}: {content}")
    
    conversations.append({
        "full_text": "\n".join(dialogue),
        "language": item.get("language"),
        "model": item.get("model")
    })

# Сохранение
json_path = os.path.join("..", "data", "processed", "wildchat_news_texts.json")
os.makedirs(os.path.dirname(json_path), exist_ok=True) 
with open(json_path, "w", encoding="utf-8") as f:
    json.dump(conversations, f, indent=2, ensure_ascii=False)

print(f"Готово! Сохранено {len(conversations)} диалогов")