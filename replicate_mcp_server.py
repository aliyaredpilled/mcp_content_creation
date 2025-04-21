import os
import replicate # Убедимся, что этот импорт есть
# import replicate # Удаляем, т.к. используется в replicate_client
import requests
from pathlib import Path
import uuid
import subprocess
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP
from replicate_client import call_replicate_api, call_replicate_video_api # <--- Импортируем новую функцию
from elevenlabs_client import call_elevenlabs_tts_api, DEFAULT_OUTPUT_FORMAT # <--- Импорт новой функции и константы
import traceback # Добавляем импорт traceback сюда, т.к. он используется в блоке except

# --- Константы и Настройки ---
# MODEL_ID = "..." # Удаляем, перенесено в replicate_client
DEFAULT_ASPECT_RATIO = "1:1" # Соотношение по умолчанию, если не указано

# Загрузка API ключа из .env
load_dotenv()
# REPLICATE_API_KEY = os.getenv("REPLICATE_API_TOKEN") # Удаляем, загрузка в replicate_client

# if not REPLICATE_API_KEY: # Удаляем проверку здесь
#     print("Ошибка: Ключ REPLICATE_API_TOKEN не найден в .env файле или переменных окружения.")
#     print("Пожалуйста, создайте файл .env и добавьте туда REPLICATE_API_TOKEN=ВАШ_КЛЮЧ")
    # exit() # Не лучшее решение для сервера

# --- Инициализация MCP Сервера ---
mcp = FastMCP("Replicate Image Generator")

# --- Вспомогательная функция для скачивания ---
def download_file(url: str, save_path: Path) -> bool:
    """Скачивает файл по URL и сохраняет его."""
    try:
        response = requests.get(url, stream=True, timeout=120) # Увеличенный таймаут
        response.raise_for_status()
        with open(save_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"Файл сохранен: {save_path}")
        return True
    except requests.exceptions.RequestException as e:
        print(f"Ошибка скачивания {url}: {e}")
        return False
    except Exception as e:
        print(f"Ошибка сохранения {save_path}: {e}")
        return False

def _try_open_file(file_path: Path):
    """Пытается открыть файл системной командой."""
    try:
        print(f"Попытка открыть: {file_path}")
        # Используем check=True, чтобы выбросить исключение при ошибке команды open
        # capture_output=True скрывает вывод команды (например, сообщения об ошибках, если 'open' не может найти приложение)
        subprocess.run(["open", str(file_path)], check=True, capture_output=True, text=True)
        print(f"Команда 'open' для {file_path} выполнена.")
        return True
    except FileNotFoundError:
        print(f"Ошибка: Команда 'open' не найдена. Не могу открыть файл автоматически (только macOS).")
    except subprocess.CalledProcessError as open_err:
        # stderr может содержать полезную информацию об ошибке от команды open
        error_details = open_err.stderr if open_err.stderr else str(open_err)
        print(f"Ошибка при выполнении команды 'open' для {file_path}: {error_details}")
    except Exception as general_err:
        print(f"Неожиданная ошибка при открытии файла {file_path}: {general_err}")
    return False

# --- Инструмент MCP для Изображений ---
@mcp.tool()
def generate_and_save_images(
    prompt: str,
    num_outputs: int,
    output_dir: str,
    # --- Необязательные параметры --- #
    lora_hf_id: str = None, # <--- Изменена аннотация типа
    lora_trigger_word: str = None, # <--- Изменена аннотация типа
    filename_prefix: str = "image",
    aspect_ratio: str = DEFAULT_ASPECT_RATIO
) -> list[str]:
    """
    Генерирует изображения с помощью Replicate, опционально используя LoRA,
    сохраняет их в локальную папку и пытается открыть. Возвращает список путей к сохраненным файлам.

    Args:
        prompt: Текстовый промпт для генерации.
        num_outputs: Количество изображений для генерации (обычно 1-4).
        output_dir: Относительный путь к папке для сохранения.
        lora_hf_id: (Опционально) Идентификатор LoRA на Hugging Face.
        lora_trigger_word: (Опционально) Ключевое слово для активации LoRA.
        filename_prefix: Префикс для имен файлов (по умолчанию "image").
        aspect_ratio: Соотношение сторон изображения (например, "1:1", "16:9", "9:16").
                      По умолчанию "1:1".

    Returns:
        Список строк с путями к успешно сохраненным файлам или список с одной строкой ошибки.
    """
    api_result = call_replicate_api(
        prompt=prompt,
        num_outputs=num_outputs,
        aspect_ratio=aspect_ratio,
        lora_hf_id=lora_hf_id,
        lora_trigger_word=lora_trigger_word
    )

    if isinstance(api_result, str): # Ошибка от API клиента
        return [api_result]

    output_urls = api_result # Теперь это точно список URL

    try:
        save_directory = Path(output_dir)
        save_directory.mkdir(parents=True, exist_ok=True)
        # print(f"Каталог для сохранения: {save_directory.resolve()}") # Убрал

        saved_files = []
        opened_files = []
        for i, url in enumerate(output_urls):
            if num_outputs > 1:
                filename = f"{filename_prefix}_{uuid.uuid4().hex[:6]}_{i+1}.png" # Добавляем уникальный ID
            else:
                filename = f"{filename_prefix}_{uuid.uuid4().hex[:6]}.png" # Добавляем уникальный ID

            save_path = save_directory / filename
            # print(f"Целевой путь сохранения: {save_path}") # Убрал

            print(f"Скачивание {url}...")
            if download_file(url, save_path):
                saved_files.append(str(save_path))
                if _try_open_file(save_path):
                    opened_files.append(str(save_path))

        print(f"Успешно сохранено {len(saved_files)} файлов.")
        if opened_files:
            print(f"Успешно открыто (или предпринята попытка): {len(opened_files)} файлов.")

        return saved_files

    except Exception as e:
        error_message = f"Ошибка при обработке/сохранении файлов: {e}"
        print(error_message)
        traceback.print_exc()
        return [error_message]

# --- НОВЫЙ Инструмент MCP для Видео ---
@mcp.tool()
def generate_and_save_video(
    prompt: str,
    output_dir: str,
    first_frame_image_path: str = None,
    filename_prefix: str = "video",
) -> list[str]:
    """
    Генерирует видео с помощью Replicate (опционально используя начальный кадр),
    сохраняет его в локальную папку и пытается открыть.
    Возвращает список с путем к сохраненному файлу или сообщение об ошибке.

    Args:
        prompt: Текстовый промпт для генерации видео (на английском).
        output_dir: Относительный путь к папке для сохранения видео.
        first_frame_image_path: (Опционально) Путь к файлу изображения для первого кадра.
        filename_prefix: Префикс для имени файла (по умолчанию "video").

    Returns:
        Список строк: содержит путь к успешно сохраненному видеофайлу
        или одно сообщение об ошибке.
    """
    api_result = call_replicate_video_api(
        prompt=prompt,
        first_frame_image_path=first_frame_image_path
    )

    if not isinstance(api_result, str) or not api_result.startswith("http"):
        error_message = f"Ошибка API или неверный URL: {api_result}"
        print(error_message)
        return [error_message]

    video_url = api_result

    try:
        save_directory = Path(output_dir)
        save_directory.mkdir(parents=True, exist_ok=True)
        # print(f"Каталог для сохранения видео: {save_directory.resolve()}") # Убрал

        filename = f"{filename_prefix}_{uuid.uuid4().hex[:6]}.mp4" # Добавляем уникальный ID
        save_path = save_directory / filename
        # print(f"Целевой путь сохранения видео: {save_path}") # Убрал

        print(f"Скачивание видео {video_url}...")
        if download_file(video_url, save_path):
            saved_file_path = str(save_path)
            print(f"Видео успешно сохранено: {saved_file_path}")
            _try_open_file(save_path) # Попытка открыть
            return [saved_file_path] # Возвращаем путь в списке
        else:
            return ["Ошибка: Не удалось скачать видео."] # Возвращаем ошибку в списке

    except Exception as e:
        error_message = f"Ошибка при обработке/сохранении видео: {e}"
        print(error_message)
        traceback.print_exc()
        return [error_message]

# --- НОВЫЙ Инструмент MCP для ElevenLabs TTS ---
@mcp.tool()
def generate_and_save_tts(
    text: str,
    output_dir: str,
    voice_id: str = None,
    filename_prefix: str = "tts",
    model_id: str = None,
    output_format: str = None
) -> list[str]:
    """
    Генерирует речь из текста с помощью ElevenLabs TTS, сохраняет аудиофайл
    в локальную папку и пытается его открыть. Использует значения по умолчанию,
    если voice_id, model_id или output_format не указаны.

    Args:
        text: Текст для озвучки.
        output_dir: Относительный путь к папке для сохранения аудио.
        voice_id: (Опционально) Идентификатор голоса ElevenLabs.
        filename_prefix: Префикс для имени файла (по умолчанию "tts").
        model_id: (Опционально) Идентификатор модели ElevenLabs.
        output_format: (Опционально) Формат вывода аудио (по умолчанию mp3_44100_128).

    Returns:
        Список строк: содержит путь к успешно сохраненному аудиофайлу
        или одно сообщение об ошибке.
    """
    # Устанавливаем формат по умолчанию, если он не передан
    actual_output_format = output_format if output_format else DEFAULT_OUTPUT_FORMAT
    file_extension = actual_output_format.split('_')[0] # Получаем расширение (mp3, pcm и т.д.)

    api_result = call_elevenlabs_tts_api(
        text=text,
        voice_id=voice_id, # Передаем None, если не указан, клиент использует свой default
        model_id=model_id, # Передаем None, если не указан, клиент использует свой default
        output_format=actual_output_format # Передаем актуальный формат
    )

    if isinstance(api_result, str): # Ошибка от API клиента
        return [api_result]

    audio_bytes = api_result # Теперь это точно байты

    try:
        save_directory = Path(output_dir)
        save_directory.mkdir(parents=True, exist_ok=True)
        # print(f"Каталог для сохранения TTS: {save_directory.resolve()}") # Убрал

        filename = f"{filename_prefix}_{uuid.uuid4().hex[:6]}.{file_extension}" # Уникальный ID и правильное расширение
        save_path = save_directory / filename
        # print(f"Целевой путь сохранения TTS: {save_path}") # Убрал

        print(f"Сохранение TTS аудио в {save_path}...")
        with open(save_path, "wb") as f:
            f.write(audio_bytes)

        saved_file_path = str(save_path)
        print(f"TTS аудио успешно сохранено: {saved_file_path}")
        _try_open_file(save_path) # Попытка открыть
        return [saved_file_path] # Возвращаем путь в списке

    except Exception as e:
        error_message = f"Ошибка при сохранении/обработке TTS аудио: {e}"
        print(error_message)
        traceback.print_exc()
        return [error_message]

# --- Запуск Сервера ---
if __name__ == "__main__":
    print("Запуск MCP Replicate Server...")
    # Здесь можно добавить параметры для mcp.run(), если нужно изменить хост/порт
    # например: mcp.run(host="0.0.0.0", port=8000)
    mcp.run()
    print("MCP Replicate Server остановлен.") 