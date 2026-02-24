# install vllm
from vllm import LLM, SamplingParams
import subprocess
import json
import time
import os
import torch
def get_gpu_memory():
    result = subprocess.run(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'], 
                           capture_output=True, text=True)
    return int(result.stdout.strip())

def main():
    # Загружаем промпт
    data_path = os.path.join("..", "prompts", "mistral_ner.txt")
    with open(data_path, "r", encoding="utf-8") as f:
        prompt_template = f.read()

    # Загружаем данные
    data_path = os.path.join("..", "data", "processed", "wildchat_news_texts.json")
    with open(data_path, "r", encoding="utf-8") as f:
        conversations = json.load(f)
    texts = [item["full_text"] for item in conversations][:100]  # первые 100

    # Подготавливаем промпты для каждого диалога
    prompts = [prompt_template.format(text=t[:2000]) for t in texts]  # обрезаем длинные

    # Инициализируем модель vLLM
    print("🔄 Загрузка модели...")
    start_load = time.time()
    llm = LLM(
        model="TheBloke/Mistral-7B-Instruct-v0.1-GPTQ",
        quantization="gptq",
        tensor_parallel_size=1,
        max_model_len=4096,
        gpu_memory_utilization=0.8,
        trust_remote_code=True,
        disable_log_stats=True,
        enforce_eager=False
        #disable_flashinfer=True
    )
    load_time = time.time() - start_load
    print(f"✅ Модель загружена за {load_time:.2f} сек")

    # Параметры генерации
    sampling_params = SamplingParams(
        temperature=0.1,
        top_p=0.95,
        max_tokens=256,
        stop=["\n\n", "```"]
    )

    print(f"\n⚡ Запуск инференса на {len(prompts)} диалогах...")
    start_inference = time.time()
    memory_before = get_gpu_memory()

    outputs = llm.generate(prompts, sampling_params)

    memory_after = get_gpu_memory()
    inference_time = time.time() - start_inference

    # Собираем результаты
    results = []
    total_tokens = 0
    for i, output in enumerate(outputs):
        generated_text = output.outputs[0].text
        total_tokens += len(output.outputs[0].token_ids)
        results.append({
            "text_preview": texts[i][:100] + "...",
            "llm_output": generated_text
        })

    # Метрики
    print("\n" + "="*50)
    print(" СТАТИСТИКА vLLM")
    print("="*50)
    print(f" Обработано диалогов: {len(results)}")
    print(f" Сгенерировано токенов: {total_tokens}")
    print(f" Средняя длина ответа: {total_tokens/len(results):.1f}")
    print(f"\n⏱ Время загрузки модели: {load_time:.2f} сек")
    print(f"\n⏱ Время инференса: {inference_time:.2f} сек")

    print(f"    Диалогов/сек: {len(texts) / inference_time:.3f}")
    print(f" Токенов/сек: {total_tokens/inference_time:.2f}")
    print(f"   Время на диалог: {inference_time/len(results)*1000:.2f} мс")
    print("="*50)
    # Использование GPU (если доступно)
    print("\n🎮 ИСПОЛЬЗОВАНИЕ GPU:")
    print(f" Память во время инференса: {memory_after} MB")
    print(f" Рост памяти: {memory_after - memory_before} MB")
    print("="*50)

    # Сохраняем
    results_dir = os.path.join("..", "results")
    os.makedirs(results_dir, exist_ok=True)  # создаст папку, если её нет

    results_path = os.path.join(results_dir, "mistral7b_output_vllm.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)


    print(f"\n💾 Результаты сохранены в results/mistral7b_output_vllm.json")
if __name__ == "__main__":
    main()