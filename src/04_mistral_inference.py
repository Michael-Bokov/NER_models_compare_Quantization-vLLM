# # pip install --upgrade bitsandbytes transformers accelerate
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch
import time
import json
from tqdm import tqdm
import os

# Настройки
model_name = "mistralai/Mistral-7B-Instruct-v0.1"
NUM_DIALOGS = 100          
MAX_NEW_TOKENS = 256       
TEMPERATURE = 0.1          

# Конфигурация 4-битного квантования
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16
)

print("🔄 Загрузка токенизатора и модели...")
start_load = time.time()
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=quant_config,
    device_map="auto",          # автоматически на GPU
    trust_remote_code=True
)
load_time = time.time() - start_load
print(f"✅ Модель загружена за {load_time:.2f} сек")

print(f"Модель размещена на: {model.device}")

# Загружаем данные
data_path = os.path.join("..", "data", "processed", "wildchat_news_texts.json")
with open(data_path, "r", encoding="utf-8") as f:
    conversations = json.load(f)

# Берём первые NUM_DIALOGS диалогов
texts = [item["full_text"] for item in conversations][:NUM_DIALOGS]
print(f"📊 Обрабатываем {len(texts)} диалогов")

# Промпт для извлечения сущностей (можно менять под свои нужды)
prompt_template = """Extract named entities from the following dialogue. 
Return them as a JSON object with keys: PERSON, ORG, LOC, EVENT, DATE, IMPACT, SOURCE.
If no entities of a type are found, use an empty list.
Only output valid JSON, no additional text.

Dialogue:
{text}

JSON output:"""

# Подготовка к инференсу
results = []
all_generated_tokens = 0

print("\n⚡ Запуск инференса...")
start_inference = time.time()

for text in tqdm(texts, desc="Обработка диалогов"):
    
    truncated_text = text[:2000]  
    prompt = prompt_template.format(text=truncated_text)
    
    # Токенизация
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to("cuda")
    
    # Генерация
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=TEMPERATURE,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    # Декодируем только новые токены (не входной промпт)
    generated_tokens = outputs[0][inputs.input_ids.shape[1]:]
    all_generated_tokens += len(generated_tokens)
    
    response = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    
    # Сохраняем результат
    results.append({
        "text_preview": truncated_text[:200] + "...",
        "llm_output": response,
        "num_tokens": len(generated_tokens)
    })

inference_time = time.time() - start_inference

# Статистика
avg_tokens_per_dialog = all_generated_tokens / len(texts)
tokens_per_sec = all_generated_tokens / inference_time
dialogs_per_sec = len(texts) / inference_time

print("\n" + "="*50)
print(" СТАТИСТИКА ОБРАБОТКИ (Mistral-7B 4bit)")
print("="*50)
print(f" Обработано диалогов: {len(texts)}")
print(f" Сгенерировано токенов: {all_generated_tokens}")
print(f" Средняя длина ответа: {avg_tokens_per_dialog:.1f} токенов")
print("\n  ВРЕМЯ ВЫПОЛНЕНИЯ:")
print(f"   Загрузка модели: {load_time:.2f} сек")
print(f"   Инференс: {inference_time:.2f} сек")
print(f"   Всего: {load_time + inference_time:.2f} сек")
print("\n ПРОИЗВОДИТЕЛЬНОСТЬ:")
print(f"    Диалогов/сек: {dialogs_per_sec:.3f}")
print(f"    Токенов/сек: {tokens_per_sec:.2f}")
print(f"    Время на диалог: {inference_time/len(texts)*1000:.2f} мс")
print("="*50)

# Использование GPU (если доступно)
if torch.cuda.is_available():
    print("\n🎮 ИСПОЛЬЗОВАНИЕ GPU:")
    memory_allocated = torch.cuda.memory_allocated(0) / 1e9
    memory_reserved = torch.cuda.memory_reserved(0) / 1e9
    print(f"   Выделено памяти: {memory_allocated:.2f} GB")
    print(f"   Зарезервировано: {memory_reserved:.2f} GB")
    print(f"   Свободно: {torch.cuda.get_device_properties(0).total_memory / 1e9 - memory_allocated:.2f} GB")
    print("="*50)

# Сохраняем результаты (необязательно)
results_path = os.path.join("..",  "results", "mistral7b_output.json")
os.makedirs(os.path.dirname(results_path), exist_ok=True)
with open(results_path, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
print(f"\n💾 Результаты сохранены в: {results_path}")

# Пример вывода для первого диалога
if results:
    print("\n Пример ответа LLM для первого диалога:")
    print(results[0]["llm_output"][:500])