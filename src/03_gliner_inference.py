from gliner import GLiNER
import os
import json
import torch
import time
from tqdm import tqdm
from datetime import timedelta

# Загружаем модель
model = GLiNER.from_pretrained("urchade/gliner_multi-v2.1")
labels = ["PERSON", "ORG", "LOC", "EVENT", "DATE", "IMPACT", "SOURCE"]
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
print(device)
# Правильный путь к данным
data_path = os.path.join("..", "data", "processed", "wildchat_news_texts.json")
abs_path = os.path.abspath(data_path)
print(f"Загружаем данные из: {abs_path}")

# Загружаем данные
conversations = json.load(open(abs_path, "r", encoding="utf-8"))
print(f"Загружено {len(conversations)} диалогов")

# Извлекаем ТОЛЬКО текст для GLiNER
texts = [item["full_text"] for item in conversations]  

# Создаем папку для результатов
os.makedirs("results", exist_ok=True)

# Инференс с прогресс-баром
results = []
start_inference = time.time()
for text in tqdm(texts, desc="Обработка диалогов"):
    ents = model.predict_entities(text, labels, threshold=0.5)  # можно играть threshold
    results.append({
        "text_preview": text[:200] + "...",  # сохраняем начало текста для контекста
        "entities": ents
    })

# Сохраняем результаты
results_path = os.path.join("..", "results", "entities_gliner.json")
with open(results_path, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print(f"Результаты сохранены в: {os.path.abspath(results_path)}")
inference_time = time.time() - start_inference

# Статистика
total_entities = sum(len(r["entities"]) for r in results)
print(f"Всего найдено сущностей: {total_entities}")
print(f"   Инференс: {inference_time:.2f} сек ({timedelta(seconds=int(inference_time))})")
print(f"В среднем на диалог: {total_entities/len(results):.2f}")