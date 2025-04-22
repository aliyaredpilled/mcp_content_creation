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
import logging
import sys # Для вывода в stderr

# --- Настройка базового логгера ---
log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
# Изменяем уровень логирования на DEBUG для большей детализации
logging.basicConfig(level=logging.DEBUG, 
                    format=log_format,
                    handlers=[
                        logging.StreamHandler(sys.stderr), # Вывод в stderr (как print)
                        logging.FileHandler("mcp_server.log"), # Опционально: запись в файл
                    ])

# Получение логгера для текущего модуля
logger = logging.getLogger(__name__)

# Можно установить разный уровень для разных библиотек, если нужно
logging.getLogger("replicate").setLevel(logging.WARNING)
logging.getLogger("elevenlabs").setLevel(logging.WARNING)
# ------------------------------------

# --- Константы и Настройки ---
# MODEL_ID = "..." # Удаляем, перенесено в replicate_client
DEFAULT_ASPECT_RATIO = "9:16" # Соотношение по умолчанию, если не указано

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
        logger.debug(f"Начало скачивания файла с URL: {url}")
        response = requests.get(url, stream=True, timeout=120) # Увеличенный таймаут
        response.raise_for_status()
        with open(save_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        logger.info(f"Файл сохранен: {save_path}")
        return True
    except requests.exceptions.RequestException as e:
        # Добавляем exc_info=True для полного трейсбека в логах
        logger.error(f"Ошибка скачивания {url}: {e}", exc_info=True)
        return False
    except Exception as e:
        logger.exception(f"Неожиданная ошибка сохранения файла {save_path}")
        return False

def _try_open_file(file_path: Path):
    """Пытается открыть файл системной командой."""
    try:
        logger.info(f"Попытка открыть: {file_path}")
        # Используем check=True, чтобы выбросить исключение при ошибке команды open
        # capture_output=True скрывает вывод команды (например, сообщения об ошибках, если 'open' не может найти приложение)
        subprocess.run(["open", str(file_path)], check=True, capture_output=True, text=True)
        logger.info(f"Команда 'open' для {file_path} выполнена.")
        return True
    except FileNotFoundError:
        logger.warning("Ошибка: Команда 'open' не найдена. Не могу открыть файл автоматически (только macOS).")
    except subprocess.CalledProcessError as open_err:
        # stderr может содержать полезную информацию об ошибке от команды open
        error_details = open_err.stderr if open_err.stderr else str(open_err)
        logger.error(f"Ошибка при выполнении команды 'open' для {file_path}: {error_details}")
    except Exception as general_err:
        logger.exception(f"Неожиданная ошибка при открытии файла {file_path}")
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
        prompt: Текстовый промпт для генерации на английском языке.
        num_outputs: Количество изображений для генерации (обычно 1-4).
        output_dir: Абсолютный путь к папке для сохранения.
        lora_hf_id: (Опционально) Идентификатор LoRA на Hugging Face.
        lora_trigger_word: (Опционально) Ключевое слово для активации LoRA.
        filename_prefix: Префикс для имен файлов (по умолчанию "image").
        aspect_ratio: Соотношение сторон изображения (например, "1:1", "16:9", "9:16"). Лучше всего 9:16!)

    Returns:
        Список строк с путями к успешно сохраненным файлам или список с одной строкой ошибки.
    """
    logger.info(
        f"Вызов generate_and_save_images: "
        f"prompt='{prompt[:50]}...', num={num_outputs}, dir='{output_dir}', "
        f"lora='{lora_hf_id}', trigger='{lora_trigger_word}', prefix='{filename_prefix}', ar='{aspect_ratio}'"
    )

    api_result = call_replicate_api(
        prompt=prompt,
        num_outputs=num_outputs,
        aspect_ratio=aspect_ratio,
        lora_hf_id=lora_hf_id,
        lora_trigger_word=lora_trigger_word
    )

    if isinstance(api_result, str):
        logger.error(f"Ошибка API Replicate при получении URL: {api_result}")
        return [api_result]

    # Используем результат вызова API, который содержит список URL
    output_urls = api_result
    logger.info(f"Получено {len(output_urls)} URL от Replicate API.")

    try:
        # Логика ниже остается для сохранения файлов по полученным URL
        save_directory = Path(output_dir)
        try:
            save_directory.mkdir(parents=True, exist_ok=True)
            logger.debug(f"Директория для сохранения: {save_directory.resolve()}")
        except OSError as e:
            logger.exception(f"Не удалось создать директорию {save_directory}")
            return [f"Ошибка создания директории: {e}"]

        saved_files = []
        opened_files = []
        for i, url in enumerate(output_urls):
            if num_outputs > 1:
                filename = f"{filename_prefix}_{i+1}.png" # Убрали хеш
            else:
                filename = f"{filename_prefix}.png" # Убрали хеш

            save_path = save_directory / filename
            logger.debug(f"Скачивание {url} в {save_path}...")

            if download_file(url, save_path):
                saved_files.append(str(save_path))
                if _try_open_file(save_path):
                    opened_files.append(str(save_path))

        logger.info(f"Успешно сохранено {len(saved_files)} файлов в {save_directory}.")
        if opened_files:
            logger.info(f"Предпринята попытка открыть: {len(opened_files)} файлов.")

        return saved_files

    except Exception as e:
        logger.exception("Неожиданная ошибка при обработке/сохранении файлов изображений")
        # Возвращаем конкретное сообщение об ошибке
        return [f"Неожиданная ошибка при обработке/сохранении изображений: {e}"]

# --- НОВЫЙ Инструмент MCP для Видео ---
@mcp.tool()
def generate_and_save_video(
    prompt: str,
    output_dir: str,
    model_name: str,
    first_frame_image_path: str | None = None,
    end_image_path: str | None = None,
    filename_prefix: str = "video",
) -> list[str]:
    """
    Генерирует видео с помощью Replicate, используя указанную модель (minimax или kling),
    опционально используя начальный кадр и/или конечный кадр (для kling).
    Сохраняет его в локальную папку и пытается открыть.
    Возвращает список с путем к сохраненному файлу или сообщение об ошибке.

    Args:
        prompt: Текстовый промпт для генерации видео (на английском).
        output_dir: Абсолютный путь к папке для сохранения видео.
        model_name: Название модели для использования ('minimax' или 'kling').
        first_frame_image_path: (Опционально) Абсолютный путь к файлу изображения для первого кадра.
        end_image_path: (Опционально) Абсолютный путь к файлу изображения для последнего кадра (только для 'kling').
        filename_prefix: Префикс для имени файла (по умолчанию "video").

    Returns:
        Список строк: содержит путь к успешно сохраненному видеофайлу
        или одно сообщение об ошибке.
    """
    logger.info(
        f"Вызов generate_and_save_video (модель: {model_name}): "
        f"prompt='{prompt[:50]}...', dir='{output_dir}', "
        f"frame='{first_frame_image_path}', end_frame='{end_image_path}', prefix='{filename_prefix}'"
    )

    # Вызываем функцию клиента, передавая все необходимые параметры
    api_result = call_replicate_video_api(
        prompt=prompt,
        model_name=model_name,
        first_frame_image_path=first_frame_image_path,
        end_image_path=end_image_path
    )

    # Проверяем, что результат - это валидный URL (строка, начинающаяся с http)
    # Эта проверка остается актуальной, т.к. call_replicate_video_api возвращает либо URL, либо строку ошибки
    if not isinstance(api_result, str) or not api_result.startswith("http"):
        error_message = f"Ошибка API Replicate или неверный URL для видео: {api_result}"
        logger.error(error_message)
        return [error_message]

    video_url = api_result
    logger.info(f"Получен URL видео от Replicate API: {video_url}")

    try:
        save_directory = Path(output_dir)
        try:
            save_directory.mkdir(parents=True, exist_ok=True)
            logger.debug(f"Директория для сохранения видео: {save_directory.resolve()}")
        except OSError as e:
            logger.exception(f"Не удалось создать директорию {save_directory}")
            return [f"Ошибка создания директории: {e}"]

        filename = f"{filename_prefix}_{uuid.uuid4().hex[:6]}.mp4" # Добавляем уникальный ID
        save_path = save_directory / filename
        logger.debug(f"Скачивание видео {video_url} в {save_path}...")

        if download_file(video_url, save_path):
            saved_file_path = str(save_path)
            logger.info(f"Видео успешно сохранено: {saved_file_path}")
            _try_open_file(save_path) # Попытка открыть
            logger.info(f"generate_and_save_video завершен успешно.")
            return [saved_file_path] # Возвращаем путь в списке
        else:
            logger.error("Не удалось скачать видеофайл.")
            return ["Ошибка: Не удалось скачать видео."] # Возвращаем ошибку в списке

    except Exception as e:
        logger.exception("Неожиданная ошибка при обработке/сохранении видео")
        # Возвращаем конкретное сообщение об ошибке
        return [f"Неожиданная ошибка при обработке/сохранении видео: {e}"]

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
        output_dir: Абсолютный путь к папке для сохранения аудио.
        voice_id: (Опционально) Идентификатор голоса ElevenLabs.
        filename_prefix: Префикс для имени файла (по умолчанию "tts").
        model_id: (Опционально) Идентификатор модели ElevenLabs.
        output_format: (Опционально) Формат вывода аудио (по умолчанию mp3_44100_128).

    Returns:
        Список строк: содержит путь к успешно сохраненному аудиофайлу
        или одно сообщение об ошибке.
    """
    logger.info(
        f"Вызов generate_and_save_tts: "
        f"text='{text[:50]}...', dir='{output_dir}', "
        f"voice='{voice_id}', model='{model_id}', format='{output_format}', prefix='{filename_prefix}'"
    )

    # Устанавливаем формат по умолчанию, если он не передан
    actual_output_format = output_format if output_format else DEFAULT_OUTPUT_FORMAT
    try:
        file_extension = actual_output_format.split('_')[0] # Получаем расширение (mp3, pcm и т.д.)
    except IndexError:
        logger.warning(f"Не удалось определить расширение из формата '{actual_output_format}'. Используется 'bin'.")
        file_extension = "bin" # Запасной вариант

    api_result = call_elevenlabs_tts_api(
        text=text,
        voice_id=voice_id, # Передаем None, если не указан, клиент использует свой default
        model_id=model_id, # Передаем None, если не указан, клиент использует свой default
        output_format=actual_output_format # Передаем актуальный формат
    )

    if isinstance(api_result, str): # Ошибка от API клиента
        logger.error(f"Ошибка API ElevenLabs: {api_result}")
        return [api_result]

    audio_bytes = api_result # Теперь это точно байты
    logger.info(f"Получено {len(audio_bytes)} байт аудио от ElevenLabs API.")

    try:
        save_directory = Path(output_dir)
        try:
            save_directory.mkdir(parents=True, exist_ok=True)
            logger.debug(f"Директория для сохранения TTS: {save_directory.resolve()}")
        except OSError as e:
            logger.exception(f"Не удалось создать директорию {save_directory}")
            return [f"Ошибка создания директории: {e}"]

        filename = f"{filename_prefix}_{uuid.uuid4().hex[:6]}.{file_extension}" # Уникальный ID и правильное расширение
        save_path = save_directory / filename
        logger.debug(f"Сохранение TTS аудио в {save_path}...")

        with open(save_path, "wb") as f:
            f.write(audio_bytes)

        saved_file_path = str(save_path)
        logger.info(f"TTS аудио успешно сохранено: {saved_file_path}")
        _try_open_file(save_path) # Попытка открыть
        logger.info(f"generate_and_save_tts завершен успешно.")
        return [saved_file_path] # Возвращаем путь в списке

    except Exception as e:
        logger.exception("Неожиданная ошибка при сохранении/обработке TTS аудио")
        # Возвращаем конкретное сообщение об ошибке
        return [f"Неожиданная ошибка при сохранении/обработке TTS аудио: {e}"]

# --- Запуск Сервера ---
if __name__ == "__main__":
    logger.info("Запуск MCP Replicate Server...")
    # Здесь можно добавить параметры для mcp.run(), если нужно изменить хост/порт
    # например: mcp.run(host="0.0.0.0", port=8000)
    try:
        mcp.run()
    except Exception as e:
        logger.critical("Критическая ошибка при запуске/работе MCP сервера", exc_info=True)
    finally:
        logger.info("MCP Replicate Server остановлен.") 