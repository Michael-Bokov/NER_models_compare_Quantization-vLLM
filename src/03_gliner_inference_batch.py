from gliner import GLiNER
import os
import json
import torch
import time
from tqdm import tqdm
from datetime import timedelta

# Настройки
BATCH_SIZE = 8          
THRESHOLD = 0.5         
labels = ["PERSON", "ORG", "LOC", "EVENT", "DATE", "IMPACT", "SOURCE"]

print(" Загрузка модели...")
start_load = time.time()
model = GLiNER.from_pretrained("urchade/gliner_multi-v2.1")
load_time = time.time() - start_load
print(f" Модель загружена за {load_time:.2f} сек")

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
print(f" Модель на: {device}")
if device == "cuda":
    print(f"   GPU: {torch.cuda.get_device_name(0)}")
    print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

# Загружаем данные
print("\n Загрузка данных...")
data_path = os.path.join("..", "data", "processed", "wildchat_news_texts.json")
with open(data_path, "r", encoding="utf-8") as f:
    conversations = json.load(f)

texts = [item["full_text"] for item in conversations]
print(f" Загружено {len(texts)} диалогов")
total_chars = sum(len(t) for t in texts)
approx_tokens = total_chars / 4
print(f" Примерно {approx_tokens:.0f} токенов")

# Подготовка к инференсу
print("\n Запуск инференса...")
start_inference = time.time()
all_results = []
total_batches = (len(texts) + BATCH_SIZE - 1) // BATCH_SIZE

for i in tqdm(range(0, len(texts), BATCH_SIZE), desc="Обработка батчей", total=total_batches):
    batch_texts = texts[i:i+BATCH_SIZE]
    
    # Вызов модели
    try:
        batch_entities = model.batch_predict_entities(batch_texts, labels, threshold=THRESHOLD)
        #print(f"Тип: {type(batch_entities)}")
        # if isinstance(batch_entities, list):
        #     print(f"Длина: {len(batch_entities)}")
        #     if len(batch_entities) > 0:
        #         print(f"Тип первого элемента: {type(batch_entities[0])}")
        #         print(f"Содержимое первого элемента: {batch_entities[0][:2] if isinstance(batch_entities[0], list) else batch_entities[0]}")
    except Exception as e:
                print(f"\n Ошибка в батче {i//BATCH_SIZE}: {e}")
                batch_entities = None
    
    # Проверка формата ответа
    if not isinstance(batch_entities, list) or len(batch_entities) != len(batch_texts):
        print(f"\n Неверный формат ответа для батча {i//BATCH_SIZE}. Создаю пустые результаты.")
        batch_entities = [[] for _ in range(len(batch_texts))]
    
    # Сохраняем результаты для каждого текста в батче
    for idx, entities in enumerate(batch_entities):
        all_results.append({
            "text_index": i + idx,
            "text_preview": batch_texts[idx][:200] + "...",
            "entities": entities,
            "num_entities": len(entities)
        })

inference_time = time.time() - start_inference

# Проверка, что результаты не пусты
if len(all_results) == 0:
    print("\n Нет результатов! Завершаю работу.")
    exit()

# Сохраняем результаты
print("\n💾 Сохранение результатов...")
results_path = os.path.join("..", "results", f"entities_gliner_batch{BATCH_SIZE}.json")
os.makedirs(os.path.dirname(results_path), exist_ok=True)
with open(results_path, "w", encoding="utf-8") as f:
    json.dump(all_results, f, indent=2, ensure_ascii=False)

# Статистика
total_entities = sum(len(r["entities"]) for r in all_results)
avg_entities = total_entities / len(all_results)

# Метрики производительности
texts_per_sec = len(texts) / inference_time
tokens_per_sec = approx_tokens / inference_time
batch_per_sec = total_batches / inference_time

# Вывод
print("\n" + "="*50)
print(" СТАТИСТИКА ОБРАБОТКИ")
print("="*50)
print(f" Обработано диалогов: {len(all_results)}")
print(f" Найдено сущностей: {total_entities}")
print(f" В среднем на диалог: {avg_entities:.2f}")
print(f" Диалогов с сущностями: {sum(1 for r in all_results if r['num_entities'] > 0)} ({sum(1 for r in all_results if r['num_entities'] > 0)/len(all_results)*100:.1f}%)")
print("\n⏱  ВРЕМЯ ВЫПОЛНЕНИЯ:")
print(f"   Загрузка модели: {load_time:.2f} сек")
print(f"   Инференс: {inference_time:.2f} сек ({timedelta(seconds=int(inference_time))})")
print(f"   Всего: {load_time + inference_time:.2f} сек")
print("\n ПРОИЗВОДИТЕЛЬНОСТЬ:")
print(f"    Диалогов/сек: {texts_per_sec:.2f}")
print(f"    Токенов/сек: {tokens_per_sec:.2f}")
print(f"    Батчей/сек: {batch_per_sec:.2f}")
print(f"    Время на диалог: {inference_time/len(texts)*1000:.2f} мс")
print("="*50)

if device == "cuda":
    print("\n🎮 ИСПОЛЬЗОВАНИЕ GPU:")
    memory_allocated = torch.cuda.memory_allocated(0) / 1e9
    memory_reserved = torch.cuda.memory_reserved(0) / 1e9
    print(f"   Выделено памяти: {memory_allocated:.2f} GB")
    print(f"   Зарезервировано: {memory_reserved:.2f} GB")
    print(f"   Свободно: {torch.cuda.get_device_properties(0).total_memory / 1e9 - memory_allocated:.2f} GB")
    print("="*50)

print(f"\n Результаты сохранены в: {results_path}")

# Покажем пример для первого диалога
if len(all_results) > 0 and len(all_results[0]["entities"]) > 0:
    print("\n Пример сущностей из первого диалога:")
    for ent in all_results[0]["entities"][:5]:
        print(f"   {ent['label']}: {ent['text']} (score: {ent['score']:.3f})")
else:
    print("\n В первом диалоге сущностей не найдено.")