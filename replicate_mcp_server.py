import os
import replicate # Убедимся, что этот импорт есть
# import replicate # Удаляем, т.к. используется в replicate_client
import requests
from pathlib import Path
import uuid
import subprocess
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP
# Возвращаем импорты инструментов
from replicate_client import call_replicate_api, call_replicate_video_api # <--- Импортируем новую функцию
from elevenlabs_client import call_elevenlabs_tts_api, DEFAULT_OUTPUT_FORMAT # <--- Импорт новой функции и константы
from openai_client import get_openai_client, call_openai_image_api # <--- Импорт новых функций OpenAI
import traceback # Добавляем импорт traceback сюда, т.к. он используется в блоке except
import logging
import sys # Для вывода в stderr
import concurrent.futures # <--- Добавляем импорт
import base64 # <--- Добавляем импорт для OpenAI
from typing import Optional, List
from openai import OpenAI

# --- Настройка базового логгера ---
log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
# Изменяем уровень логирования на DEBUG для большей детализации
logging.basicConfig(level=logging.DEBUG, 
                    format=log_format,
                    handlers=[
                        # logging.StreamHandler(sys.stderr), # Вывод в stderr (как print)
                        # logging.FileHandler("mcp_server.log"), # Опционально: запись в файл
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
# Убираем явное указание порта
mcp = FastMCP("Replicate Image Generator")

# --- Возвращаем вспомогательные функции и инструменты --- 

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
    first_frame_image_path: str = None, # <--- Убираем | None
    end_image_path: str = None, # <--- Убираем | None
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

# --- НОВЫЙ Инструмент MCP для TTS ---
@mcp.tool()
def generate_and_save_tts(
    text: str,
    output_dir: str,
    voice_id: str = None, # <--- Уже было без | None, оставляем
    filename_prefix: str = "tts",
    model_id: str = None, # <--- Уже было без | None, оставляем
    output_format: str = None # <--- Уже было без | None, оставляем
) -> list[str]:
    """
    Генерирует речь с помощью ElevenLabs, сохраняет ее в локальную папку
    и пытается открыть. Возвращает список с путем к сохраненному файлу или сообщение об ошибке.

    Args:
        text: Текст для озвучки.
        output_dir: Абсолютный путь к папке для сохранения аудио.
        voice_id: (Опционально) Идентификатор голоса ElevenLabs.
        filename_prefix: Префикс для имени файла (по умолчанию "tts").
        model_id: (Опционально) Идентификатор модели ElevenLabs.
        output_format: (Опционально) Формат аудиофайла (например, 'mp3_44100_128', 'pcm_16000').
                       По умолчанию используется DEFAULT_OUTPUT_FORMAT из elevenlabs_client.

    Returns:
        Список строк: содержит путь к успешно сохраненному аудиофайлу
        или одно сообщение об ошибке.
    """
    logger.info(
        f"Вызов generate_and_save_tts: "
        f"voice='{voice_id}', model='{model_id}', dir='{output_dir}', "
        f"prefix='{filename_prefix}', format='{output_format or DEFAULT_OUTPUT_FORMAT}', text='{text[:50]}...'"
    )

    # Используем формат по умолчанию из клиента, если не передан явно
    actual_output_format = output_format if output_format else DEFAULT_OUTPUT_FORMAT
    file_extension = actual_output_format.split("_")[0] # Извлекаем расширение (mp3, pcm, etc.)

    # Вызываем функцию TTS клиента
    api_result = call_elevenlabs_tts_api(
        text=text,
        voice_id=voice_id, # Передаем None, если не указан
        model_id=model_id, # Передаем None, если не указан
        output_format=actual_output_format
    )

    # Проверяем, что результат - это байты
    if not isinstance(api_result, bytes):
        error_message = f"Ошибка API ElevenLabs или неверный тип данных: {api_result}" # api_result уже строка ошибки
        logger.error(error_message)
        return [error_message]

    audio_bytes = api_result
    logger.info(f"Получено {len(audio_bytes)} байт аудио от ElevenLabs API.")

    try:
        save_directory = Path(output_dir)
        try:
            save_directory.mkdir(parents=True, exist_ok=True)
            logger.debug(f"Директория для сохранения TTS: {save_directory.resolve()}")
        except OSError as e:
            logger.exception(f"Не удалось создать директорию {save_directory}")
            return [f"Ошибка создания директории: {e}"]

        filename = f"{filename_prefix}_{uuid.uuid4().hex[:6]}.{file_extension}" # Используем правильное расширение
        save_path = save_directory / filename
        logger.debug(f"Сохранение TTS аудио в {save_path}...")

        # Сохраняем байты в файл
        with open(save_path, "wb") as f:
            f.write(audio_bytes)

        saved_file_path = str(save_path)
        logger.info(f"TTS аудио успешно сохранено: {saved_file_path}")
        _try_open_file(save_path) # Попытка открыть
        logger.info(f"generate_and_save_tts завершен успешно.")
        return [saved_file_path] # Возвращаем путь в списке

    except Exception as e:
        logger.exception("Неожиданная ошибка при сохранении TTS аудиофайла")
        return [f"Неожиданная ошибка при сохранении TTS: {e}"] # Возвращаем ошибку в списке

# --- Вспомогательная функция для параллельной обработки видео ---
def _process_single_video_request(request_data: dict, request_index: int, model_name: str, save_directory: Path) -> str:
    """
    Обрабатывает один запрос на генерацию видео.
    Вызывается в отдельном потоке.

    Args:
        request_data: Словарь с данными запроса ('prompt', 'first_frame_image_path', ...).
        request_index: Порядковый номер запроса (для логирования и имен файлов).
        model_name: Имя модели Replicate.
        save_directory: Путь к директории для сохранения.

    Returns:
        Строка с путем к сохраненному файлу или сообщение об ошибке.
    """
    logger.info(f"Начало обработки запроса {request_index} в потоке...")

    # Извлечение параметров
    prompt = request_data.get("prompt")
    first_frame_path = request_data.get("first_frame_image_path")
    end_frame_path = request_data.get("end_image_path")
    output_filename_base = request_data.get("output_filename")

    # Проверки обязательных полей (уже сделаны в основной функции, но можно добавить для надежности)
    if not prompt or not first_frame_path:
         # Эта ситуация не должна возникать из-за проверок выше, но на всякий случай
         error_msg = f"Запрос {request_index}: Внутренняя ошибка - отсутствуют prompt или first_frame_path."
         logger.error(error_msg)
         return error_msg
    if not Path(first_frame_path).is_file(): # Проверка файла здесь тоже важна
        error_msg = f"Запрос {request_index}: Файл первого кадра не найден: {first_frame_path}"
        logger.error(error_msg)
        return error_msg

    logger.info(
        f"  Запрос {request_index}: Параметры: prompt='{prompt[:30]}...', "
        f"frame='{first_frame_path}', end_frame='{end_frame_path}'"
    )

    try:
        # --- Вызов API --- (Эта часть выполняется в потоке)
        api_result = call_replicate_video_api(
            prompt=prompt,
            model_name=model_name,
            first_frame_image_path=first_frame_path,
            end_image_path=end_frame_path
        )

        if not isinstance(api_result, str) or not api_result.startswith("http"):
            error_message = f"Ошибка API Replicate для запроса {request_index}: {api_result}"
            logger.error(f"  {error_message}") # Добавим отступ для логов потока
            return error_message # Возвращаем ошибку из потока

        video_url = api_result
        logger.info(f"  Запрос {request_index}: получен URL видео: {video_url}")

        # --- Скачивание и сохранение --- (Эта часть тоже выполняется в потоке)
        if output_filename_base:
            if '.' in output_filename_base:
                output_filename_base = os.path.splitext(output_filename_base)[0]
            filename = f"{output_filename_base}.mp4"
        else:
            filename = f"video_{request_index}_{uuid.uuid4().hex[:6]}.mp4"

        save_path = save_directory / filename
        logger.debug(f"  Запрос {request_index}: скачивание видео в {save_path}...")

        if download_file(video_url, save_path):
            saved_file_path = str(save_path)
            logger.info(f"  Запрос {request_index}: видео успешно сохранено: {saved_file_path}")
            _try_open_file(save_path) # Попытка открыть (тоже в потоке)
            logger.info(f"Обработка запроса {request_index} в потоке завершена успешно.")
            return saved_file_path # Возвращаем путь из потока
        else:
            error_msg = f"Ошибка скачивания для запроса {request_index}. URL: {video_url}"
            logger.error(f"  {error_msg}")
            return error_msg # Возвращаем ошибку скачивания из потока

    except Exception as e:
        error_msg = f"Неожиданная ошибка при обработке запроса {request_index} в потоке: {e}"
        logger.exception(f"  {error_msg}") # Логируем с трассировкой
        return error_msg # Возвращаем неожиданную ошибку из потока

# --- ИЗМЕНЕННЫЙ Инструмент MCP для Нескольких Видео (Параллельный) ---
@mcp.tool()
def generate_and_save_multiple_videos(
    video_requests: list[dict],
    base_output_dir: str,
    model_name: str, # <-- Единая модель
    max_workers: int = 8 # <-- Макс. кол-во параллельных потоков
) -> list[str]:
    """
    Генерирует НЕСКОЛЬКО видео ПАРАЛЛЕЛЬНО с помощью Replicate, используя ОДНУ указанную модель
    на основе списка запросов, сохраняет их в базовую директорию и пытается открыть.
    Возвращает список путей к сохраненным файлам и/или сообщения об ошибках.

    Args:
        video_requests: Список словарей. Каждый словарь представляет один запрос
                        и должен содержать ключи:
                        - 'prompt' (str): Текстовый промпт для генерации.
                        - 'first_frame_image_path' (str): Абсолютный путь к файлу изображения для ПЕРВОГО кадра (ОБЯЗАТЕЛЬНО).
                        Опциональные ключи:
                        - 'end_image_path' (str): Абсолютный путь к файлу изображения для ПОСЛЕДНЕГО кадра (только для 'kling').
                        - 'output_filename' (str): Желаемое имя файла (без расширения).
                                                   Если не указано, будет сгенерировано.
        base_output_dir: Абсолютный путь к базовой папке для сохранения всех видео.
        model_name: Название модели для использования ('minimax' или 'kling'), применяется ко ВСЕМ запросам. лучше клинг!)
        max_workers: Максимальное количество потоков для параллельной обработки (по умолчанию 8).

    Returns:
        Список строк: содержит пути к успешно сохраненным видеофайлам
        и/или сообщения об ошибках для каждого запроса.
    """
    logger.info(
        f"Вызов generate_and_save_multiple_videos (ПАРАЛЛЕЛЬНО, max_workers={max_workers}): "
        f"{len(video_requests)} запросов, модель='{model_name}', base_dir='{base_output_dir}'"
    )

    results = []
    processed_count = 0
    error_count = 0

    # --- Валидация входных данных (остается) ---
    if not isinstance(video_requests, list):
        error_msg = "Ошибка: 'video_requests' должен быть списком."
        logger.error(error_msg)
        return [error_msg]
    if not base_output_dir:
        error_msg = "Ошибка: 'base_output_dir' не может быть пустым."
        logger.error(error_msg)
        return [error_msg]
    if not model_name:
         error_msg = "Ошибка: 'model_name' не может быть пустым."
         logger.error(error_msg)
         return [error_msg]

    # --- Предварительная проверка запросов (валидность структуры и обязательных полей) ---
    valid_requests_with_indices = []
    for i, req_data in enumerate(video_requests):
        request_index = i + 1
        if not isinstance(req_data, dict):
            error_msg = f"Ошибка запроса {request_index}: Элемент в 'video_requests' должен быть словарем."
            logger.error(error_msg)
            results.append(error_msg) # Добавляем ошибку сразу
            error_count += 1
            continue
        prompt = req_data.get("prompt")
        first_frame_path = req_data.get("first_frame_image_path")
        if not prompt or not first_frame_path:
            error_msg = f"Ошибка запроса {request_index}: Отсутствуют обязательные ключи 'prompt' или 'first_frame_path'."
            logger.error(error_msg)
            results.append(error_msg)
            error_count += 1
            continue
        # Файл проверим внутри потока, чтобы не блокировать основной поток слишком долго
        valid_requests_with_indices.append((req_data, request_index))

    if not valid_requests_with_indices:
        logger.warning("Нет валидных запросов для обработки.")
        # Возвращаем ошибки валидации, если они были
        return results if results else ["Нет валидных запросов для обработки."]

    # --- Создание базовой директории (остается) ---
    try:
        save_directory = Path(base_output_dir)
        save_directory.mkdir(parents=True, exist_ok=True)
        logger.debug(f"Базовая директория для сохранения видео: {save_directory.resolve()}")
    except OSError as e:
        error_msg = f"Ошибка создания базовой директории {save_directory}: {e}"
        logger.exception(error_msg)
        results.insert(0, error_msg) # Добавляем в начало списка ошибок
        return results
    except Exception as e:
        error_msg = f"Неожиданная ошибка при создании директории {base_output_dir}: {e}"
        logger.exception(error_msg)
        results.insert(0, error_msg)
        return results

    # --- Параллельная обработка запросов --- #
    futures = []
    # Используем ThreadPoolExecutor для I/O-bound задач (ожидание API, скачивание)
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        logger.info(f"Отправка {len(valid_requests_with_indices)} валидных запросов в ThreadPoolExecutor...")
        for req_data, req_index in valid_requests_with_indices:
            # Отправляем задачу на выполнение в потоке
            future = executor.submit(
                _process_single_video_request,
                req_data,         # Данные запроса
                req_index,        # Номер запроса
                model_name,       # Общее имя модели
                save_directory    # Общая папка сохранения
            )
            futures.append(future)

        logger.info("Ожидание завершения обработки запросов...")
        # Собираем результаты по мере завершения
        for future in concurrent.futures.as_completed(futures):
            try:
                result = future.result() # Получаем результат из потока (путь или строка ошибки)
                results.append(result)
                # Обновляем счетчики на основе результата
                if isinstance(result, str) and result.startswith(("Ошибка", "Неожиданная")):
                    error_count += 1
                else:
                    # Предполагаем, что строка без 'Ошибка' - это путь к файлу
                    processed_count += 1
            except Exception as e:
                # Если сама future завершилась с необработанным исключением (не должно происходить из-за try/except в _process_single_video_request)
                error_msg = f"Критическая ошибка при получении результата из потока: {e}"
                logger.exception(error_msg)
                results.append(error_msg)
                error_count += 1

    logger.info(
        f"generate_and_save_multiple_videos (ПАРАЛЛЕЛЬНО) завершен. "
        f"Успешно обработано: {processed_count}, с ошибками: {error_count}."
    )
    return results

# --- Вспомогательная функция для параллельной обработки ИЗОБРАЖЕНИЙ ---
def _process_single_image_request(request_data: dict, request_index: int, save_directory: Path) -> list[str] | str:
    """
    Обрабатывает один запрос на генерацию изображений (возможно, нескольких).
    Вызывается в отдельном потоке.

    Args:
        request_data: Словарь с данными запроса ('prompt', 'num_outputs', ...).
        request_index: Порядковый номер запроса (для логирования и имен файлов).
        save_directory: Путь к директории для сохранения.

    Returns:
        Список строк с путями к сохраненным файлам или одна строка с сообщением об ошибке.
    """
    logger.info(f"Начало обработки запроса изображений {request_index} в потоке...")

    # Извлечение параметров
    prompt = request_data.get("prompt")
    num_outputs = request_data.get("num_outputs", 1) # По умолчанию 1
    aspect_ratio = request_data.get("aspect_ratio", DEFAULT_ASPECT_RATIO) # Используем дефолт
    lora_hf_id = request_data.get("lora_hf_id")
    lora_trigger_word = request_data.get("lora_trigger_word")
    filename_prefix = request_data.get("output_filename_prefix", f"image_{request_index}") # Дефолтный префикс

    # Проверки обязательных полей
    if not prompt:
         error_msg = f"Запрос {request_index}: Отсутствует обязательный ключ 'prompt'."
         logger.error(error_msg)
         return error_msg
    if not isinstance(num_outputs, int) or num_outputs <= 0:
        num_outputs = 1
        logger.warning(f"Запрос {request_index}: Некорректное значение num_outputs, установлено в 1.")

    logger.info(
        f"  Запрос {request_index}: Параметры: prompt='{prompt[:30]}...', N={num_outputs}, AR={aspect_ratio}, "
        f"lora='{lora_hf_id}', trigger='{lora_trigger_word}', prefix='{filename_prefix}'"
    )

    saved_files_for_request = []
    try:
        # --- Вызов API --- (Эта часть выполняется в потоке)
        api_result = call_replicate_api(
            prompt=prompt,
            num_outputs=num_outputs,
            aspect_ratio=aspect_ratio,
            lora_hf_id=lora_hf_id,
            lora_trigger_word=lora_trigger_word
        )

        # call_replicate_api возвращает список URL или строку ошибки
        if isinstance(api_result, str):
            error_message = f"Ошибка API Replicate для запроса изображений {request_index}: {api_result}"
            logger.error(f"  {error_message}")
            return error_message # Возвращаем ошибку из потока

        output_urls = api_result
        logger.info(f"  Запрос {request_index}: получено {len(output_urls)} URL изображений.")

        # --- Скачивание и сохранение каждого изображения --- (Эта часть тоже выполняется в потоке)
        opened_files_count = 0
        download_errors = 0
        for i, url in enumerate(output_urls):
            if len(output_urls) > 1:
                filename = f"{filename_prefix}_{i+1}.png"
            else:
                filename = f"{filename_prefix}.png"

            save_path = save_directory / filename
            logger.debug(f"  Запрос {request_index}, Изображение {i+1}: скачивание {url} в {save_path}...")

            if download_file(url, save_path):
                saved_file_path = str(save_path)
                logger.info(f"  Запрос {request_index}, Изображение {i+1}: успешно сохранено: {saved_file_path}")
                saved_files_for_request.append(saved_file_path)
                if _try_open_file(save_path):
                    opened_files_count += 1
            else:
                logger.error(f"  Запрос {request_index}, Изображение {i+1}: ошибка скачивания URL: {url}")
                download_errors += 1

        if download_errors > 0:
             # Если были ошибки скачивания, но хотя бы что-то скачалось, вернем пути
             # Если ничего не скачалось, вернем общую ошибку
             if not saved_files_for_request:
                 return f"Ошибка скачивания всех изображений для запроса {request_index}."
             else:
                 logger.warning(f"Запрос {request_index}: Были ошибки при скачивании {download_errors} изображений.")

        if opened_files_count > 0:
             logger.info(f"  Запрос {request_index}: Предпринята попытка открыть {opened_files_count} файлов.")

        logger.info(f"Обработка запроса изображений {request_index} в потоке завершена. Сохранено: {len(saved_files_for_request)}.")
        return saved_files_for_request # Возвращаем список путей

    except Exception as e:
        error_msg = f"Неожиданная ошибка при обработке запроса изображений {request_index} в потоке: {e}"
        logger.exception(f"  {error_msg}") # Логируем с трассировкой
        return error_msg # Возвращаем неожиданную ошибку из потока

# --- НОВЫЙ Инструмент MCP для Нескольких ИЗОБРАЖЕНИЙ (Параллельный) ---
@mcp.tool()
def generate_and_save_multiple_images(
    image_requests: list[dict],
    base_output_dir: str,
    max_workers: int = 8 # <-- Макс. кол-во параллельных потоков
) -> list[str]:
    """
    Генерирует НЕСКОЛЬКО наборов изображений ПАРАЛЛЕЛЬНО с помощью Replicate
    на основе списка запросов, сохраняет их в базовую директорию и пытается открыть.
    Возвращает плоский список путей ко всем успешно сохраненным файлам и/или сообщения об ошибках.

    Args:
        image_requests: Список словарей. Каждый словарь представляет один запрос
                        и должен содержать ключи:
                        - 'prompt' (str): Текстовый промпт для генерации.
                        Опциональные ключи:
                        - 'num_outputs' (int): Количество изображений (по умолчанию 1).
                        - 'aspect_ratio' (str): Соотношение сторон (по умолчанию DEFAULT_ASPECT_RATIO).
                        - 'lora_hf_id' (str): Идентификатор LoRA.
                        - 'lora_trigger_word' (str): Ключевое слово LoRA.
                        - 'output_filename_prefix' (str): Префикс для имен файлов этого запроса
                                                          (по умолчанию "image_<request_index>").
        base_output_dir: Абсолютный путь к базовой папке для сохранения всех изображений.
        max_workers: Максимальное количество потоков для параллельной обработки (по умолчанию 8).

    Returns:
        Плоский список строк: содержит пути ко ВСЕМ успешно сохраненным файлам
        из ВСЕХ запросов и/или сообщения об ошибках для каждого запроса.
    """
    logger.info(
        f"Вызов generate_and_save_multiple_images (ПАРАЛЛЕЛЬНО, max_workers={max_workers}): "
        f"{len(image_requests)} запросов, base_dir='{base_output_dir}'"
    )

    all_results = [] # Собираем ВСЕ результаты (пути и ошибки)
    processed_files_count = 0
    error_requests_count = 0

    # --- Валидация входных данных ---
    if not isinstance(image_requests, list):
        error_msg = "Ошибка: 'image_requests' должен быть списком."
        logger.error(error_msg)
        return [error_msg]
    if not base_output_dir:
        error_msg = "Ошибка: 'base_output_dir' не может быть пустым."
        logger.error(error_msg)
        return [error_msg]

    # --- Предварительная проверка запросов ---
    valid_requests_with_indices = []
    for i, req_data in enumerate(image_requests):
        request_index = i + 1
        if not isinstance(req_data, dict):
            error_msg = f"Ошибка запроса {request_index}: Элемент в 'image_requests' должен быть словарем."
            logger.error(error_msg)
            all_results.append(error_msg)
            error_requests_count += 1
            continue
        prompt = req_data.get("prompt")
        if not prompt:
            error_msg = f"Ошибка запроса {request_index}: Отсутствует обязательный ключ 'prompt'."
            logger.error(error_msg)
            all_results.append(error_msg)
            error_requests_count += 1
            continue
        valid_requests_with_indices.append((req_data, request_index))

    if not valid_requests_with_indices:
        logger.warning("Нет валидных запросов изображений для обработки.")
        return all_results if all_results else ["Нет валидных запросов изображений для обработки."]

    # --- Создание базовой директории ---
    try:
        save_directory = Path(base_output_dir)
        save_directory.mkdir(parents=True, exist_ok=True)
        logger.debug(f"Базовая директория для сохранения изображений: {save_directory.resolve()}")
    except OSError as e:
        error_msg = f"Ошибка создания базовой директории {save_directory}: {e}"
        logger.exception(error_msg)
        all_results.insert(0, error_msg)
        return all_results
    except Exception as e:
        error_msg = f"Неожиданная ошибка при создании директории {base_output_dir}: {e}"
        logger.exception(error_msg)
        all_results.insert(0, error_msg)
        return all_results

    # --- Параллельная обработка запросов --- #
    futures = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        logger.info(f"Отправка {len(valid_requests_with_indices)} валидных запросов изображений в ThreadPoolExecutor...")
        for req_data, req_index in valid_requests_with_indices:
            future = executor.submit(
                _process_single_image_request,
                req_data,
                req_index,
                save_directory
            )
            futures.append(future)

        logger.info("Ожидание завершения обработки запросов изображений...")
        for future in concurrent.futures.as_completed(futures):
            try:
                result = future.result() # Получаем результат (список путей или строка ошибки)
                if isinstance(result, list):
                    all_results.extend(result) # Добавляем список путей
                    processed_files_count += len(result)
                elif isinstance(result, str):
                    all_results.append(result) # Добавляем строку ошибки
                    error_requests_count += 1
                else:
                    # Неожиданный тип результата
                    error_msg = f"Неожиданный тип результата из потока обработки изображений: {type(result)}"
                    logger.error(error_msg)
                    all_results.append(error_msg)
                    error_requests_count += 1
            except Exception as e:
                error_msg = f"Критическая ошибка при получении результата из потока изображений: {e}"
                logger.exception(error_msg)
                all_results.append(error_msg)
                error_requests_count += 1

    logger.info(
        f"generate_and_save_multiple_images (ПАРАЛЛЕЛЬНО) завершен. "
        f"Всего сохранено файлов: {processed_files_count}, запросов с ошибками: {error_requests_count}."
    )
    return all_results

# --- Вспомогательная функция для сохранения байтов --- (для OpenAI)
def save_image_bytes(image_bytes: bytes, save_path: Path) -> bool:
    """Сохраняет байты изображения в файл."""
    try:
        logger.debug(f"Начало сохранения {len(image_bytes)} байт в {save_path}")
        with open(save_path, "wb") as f:
            f.write(image_bytes)
        logger.info(f"Файл сохранен: {save_path}")
        return True
    except Exception as e:
        logger.exception(f"Неожиданная ошибка сохранения файла {save_path}")
        return False

# --- Вспомогательная функция для параллельной обработки OpenAI ИЗОБРАЖЕНИЙ --- (Должна быть здесь)
def _process_single_openai_image_request(
    client: OpenAI, # Передаем готовый клиент
    request_data: dict,
    request_index: int,
    save_directory: Path
) -> list[str] | str:
    """
    Обрабатывает один запрос на генерацию/редактирование OpenAI изображений.
    Вызывается в отдельном потоке.
    Если режим - редактирование и num_outputs > 1, выполняет num_outputs параллельных запросов.

    Args:
        client: Инициализированный клиент OpenAI.
        request_data: Словарь с данными запроса ('prompt', 'num_outputs', 'reference_image_paths'...).
        request_index: Порядковый номер запроса.
        save_directory: Путь к директории для сохранения.

    Returns:
        Список строк с путями к сохраненным файлам или одна строка с сообщением об ошибке.
    """
    logger.info(f"Начало обработки OpenAI запроса {request_index} в потоке...")

    # Извлечение параметров
    prompt = request_data.get("prompt")
    num_outputs = request_data.get("num_outputs", 1)
    size = request_data.get("size", "1024x1536") # Используем портретный по умолчанию
    quality = request_data.get("quality", "medium")
    reference_image_paths = request_data.get("reference_image_paths") # Может быть None
    filename_prefix = request_data.get("output_filename_prefix", f"openai_image_{request_index}")

    if not prompt:
         error_msg = f"OpenAI Запрос {request_index}: Отсутствует обязательный ключ 'prompt'."
         logger.error(error_msg)
         return error_msg

    is_editing_mode = bool(reference_image_paths)
    mode = "редактирования" if is_editing_mode else "генерации"
    logger.info(
        f"  OpenAI Запрос {request_index} (режим {mode}): Параметры: prompt='{prompt[:30]}...', "
        f"N={num_outputs}, Size={size}, Quality={quality}, Refs={len(reference_image_paths) if reference_image_paths else 0}, "
        f"Prefix='{filename_prefix}'"
    )

    all_saved_files_for_request = []
    all_api_results_base64 = [] # Собираем base64 данные или ошибки
    processing_errors = [] # Собираем ошибки обработки (сохранение, декодирование)

    try:
        # --- Вызов API ---
        # Если режим редактирования и нужно > 1 результата, делаем параллельные вызовы
        if is_editing_mode and num_outputs > 1:
            logger.info(f"  OpenAI Запрос {request_index}: Запуск {num_outputs} параллельных запросов на редактирование...")
            edit_futures = []
            # Используем вложенный ThreadPoolExecutor для параллельных API вызовов *внутри* одного запроса
            # max_workers можно ограничить, чтобы не перегружать API/сеть
            with concurrent.futures.ThreadPoolExecutor(max_workers=min(num_outputs, 4)) as edit_executor:
                for i in range(num_outputs):
                    # Каждый вызов делается с n=1 (в call_openai_image_api это учтено)
                    future = edit_executor.submit(
                        call_openai_image_api,
                        client=client,
                        prompt=prompt,
                        n=1, # Важно: n=1 для каждого вызова редактирования
                        size=size,
                        quality=quality,
                        reference_image_paths=reference_image_paths
                    )
                    edit_futures.append(future)

                # Собираем результаты параллельных вызовов
                for i, future in enumerate(concurrent.futures.as_completed(edit_futures)):
                    try:
                        result = future.result() # Список base64 (обычно 1) или строка ошибки
                        if isinstance(result, list) and result:
                            all_api_results_base64.append(result[0]) # Берем первый (и единственный) элемент
                        elif isinstance(result, str):
                            logger.error(f"  OpenAI Запрос {request_index}, Вызов {i+1}: Ошибка API: {result}")
                            # Добавляем ошибку, чтобы сообщить пользователю
                            processing_errors.append(f"Ошибка API при редактировании (вызов {i+1}): {result}")
                        else:
                            logger.error(f"  OpenAI Запрос {request_index}, Вызов {i+1}: Неожиданный результат API: {result}")
                            processing_errors.append(f"Неожиданный результат API при редактировании (вызов {i+1})")
                    except Exception as e:
                        logger.exception(f"  OpenAI Запрос {request_index}, Вызов {i+1}: Исключение при получении результата из потока редактирования")
                        processing_errors.append(f"Ошибка при получении результата редактирования (вызов {i+1}): {e}")

            logger.info(f"  OpenAI Запрос {request_index}: Завершено {num_outputs} параллельных запросов на редактирование. Получено результатов/ошибок: {len(all_api_results_base64)}/{len(processing_errors)}")

        # Иначе (генерация или редактирование с num_outputs=1), делаем один вызов
        else:
            api_result = call_openai_image_api(
                client=client,
                prompt=prompt,
                n=num_outputs if not is_editing_mode else 1, # n=1 для одного редактирования
                size=size,
                quality=quality,
                reference_image_paths=reference_image_paths
            )

            if isinstance(api_result, str): # Ошибка API
                error_message = f"Ошибка OpenAI API для запроса {request_index}: {api_result}"
                logger.error(f"  {error_message}")
                return error_message # Возвращаем сразу, если основной вызов не удался
            else:
                all_api_results_base64 = api_result # Список base64 строк

        logger.info(f"  OpenAI Запрос {request_index}: получено {len(all_api_results_base64)} base64 строк для сохранения.")

        # --- Декодирование, сохранение и открытие --- (Общее для всех режимов)
        opened_files_count = 0
        decode_errors = 0
        save_errors = 0

        for i, b64_data in enumerate(all_api_results_base64):
            unique_id = uuid.uuid4().hex[:6]
            # Нумеруем файлы последовательно, даже если они от разных API вызовов
            filename = f"{filename_prefix}_{i+1}_{unique_id}.png"
            save_path = save_directory / filename

            # Декодирование
            try:
                logger.debug(f"  Запрос {request_index}, Изобр. {i+1}: Декодирование base64...")
                image_bytes = base64.b64decode(b64_data)
            except (base64.binascii.Error, ValueError) as decode_err:
                logger.error(f"  Запрос {request_index}, Изобр. {i+1}: Ошибка декодирования base64: {decode_err}")
                decode_errors += 1
                processing_errors.append(f"Ошибка декодирования изобр. {i+1}")
                continue

            # Сохранение
            logger.debug(f"  Запрос {request_index}, Изобр. {i+1}: Сохранение в {save_path}...")
            if save_image_bytes(image_bytes, save_path):
                saved_file_path = str(save_path)
                logger.info(f"  Запрос {request_index}, Изобр. {i+1}: Успешно сохранено: {saved_file_path}")
                all_saved_files_for_request.append(saved_file_path)
                if _try_open_file(save_path):
                    opened_files_count += 1
            else:
                 logger.error(f"  Запрос {request_index}, Изобр. {i+1}: Ошибка сохранения файла.")
                 save_errors += 1
                 processing_errors.append(f"Ошибка сохранения изобр. {i+1}")

        # --- Обработка ошибок и финальный результат ---
        if decode_errors > 0:
             logger.warning(f"  Запрос {request_index}: Было {decode_errors} ошибок декодирования.")
        if save_errors > 0:
             logger.warning(f"  Запрос {request_index}: Было {save_errors} ошибок сохранения.")
        if opened_files_count > 0:
             logger.info(f"  Запрос {request_index}: Предпринята попытка открыть {opened_files_count} файлов.")

        # Собираем финальный результат: сначала пути, потом ошибки обработки/API
        final_result = all_saved_files_for_request + processing_errors

        logger.info(f"Обработка OpenAI запроса {request_index} в потоке завершена. Сохранено: {len(all_saved_files_for_request)}. Ошибок обработки/API: {len(processing_errors)}.")

        if not final_result:
             # Если нет ни путей, ни ошибок (например, API вернул пустой список)
             return f"Замечание: Для OpenAI запроса {request_index} не было получено или сохранено изображений."
        else:
             return final_result # Возвращаем список путей и/или строк ошибок

    except Exception as e:
        error_msg = f"Неожиданная ошибка при обработке OpenAI запроса {request_index} в потоке: {e}"
        logger.exception(f"  {error_msg}")
        # Возвращаем ошибку + все, что успели сохранить до этого
        return all_saved_files_for_request + [error_msg]

# --- Инструмент MCP для Нескольких OpenAI ИЗОБРАЖЕНИЙ (Параллельный) ---
@mcp.tool()
def generate_and_save_multiple_openai_images(
    image_requests: list[dict],
    base_output_dir: str,
    max_workers: int = 8
) -> list[str]:
    """
    Генерирует или редактирует НЕСКОЛЬКО наборов изображений ПАРАЛЛЕЛЬНО с помощью OpenAI (gpt-image-1)
    на основе списка запросов, сохраняет их в базовую директорию и пытается открыть.
    Возвращает плоский список путей ко всем успешно сохраненным файлам и/или сообщения об ошибках.

    **Для генерации одного изображения:** передайте список `image_requests` с одним элементом.

    Args:
        image_requests: Список словарей. Каждый словарь представляет один запрос
                        и должен содержать ключ 'prompt' (str).
                        Опциональные ключи:
                        - 'num_outputs' (int): Кол-во изображений (по умолч. 1, игнор. при ред.).
                        - 'size' (str): Размер ('1024x1024', '1536x1024', '1024x1536', 'auto', по умолч. '1024x1536').
                        - 'quality' (str): Качество ('low', 'medium', 'high', 'auto', по умолч. 'medium').
                        - 'output_filename_prefix' (str): Префикс имен файлов (по умолч. "openai_image_<index>").
                        - 'reference_image_paths' (list[str] | None): Список путей к референсам для режима редактирования.
                          При использовании нескольких референсов, рекомендуется указывать в промпте, какой референс за что отвечает, используя их описания или концептуальные имена (например, "Generate the monster (Grizzya) in the artistic style of the cat image").
        base_output_dir: Абсолютный путь к базовой папке для сохранения всех изображений.
        max_workers: Максимальное количество потоков для параллельной обработки (по умолчанию 8).

    Returns:
        Плоский список строк: содержит пути ко ВСЕМ успешно сохраненным файлам
        из ВСЕХ запросов и/или сообщения об ошибках для каждого запроса.
    """
    logger.info(
        f"Вызов generate_and_save_multiple_openai_images (ПАРАЛЛЕЛЬНО, max_workers={max_workers}): "
        f"{len(image_requests)} запросов, base_dir='{base_output_dir}'"
    )

    all_results = []
    processed_files_count = 0
    error_requests_count = 0

    # --- Валидация --- #
    if not isinstance(image_requests, list):
        return ["Ошибка: 'image_requests' должен быть списком."]
    if not base_output_dir:
        return ["Ошибка: 'base_output_dir' не может быть пустым."]

    # --- Инициализация клиента OpenAI (ОДИН РАЗ) --- #
    logger.debug("Инициализация клиента OpenAI для параллельной обработки...")
    client = get_openai_client()
    if not client:
        error_msg = "Ошибка: Не удалось инициализировать клиент OpenAI для параллельной обработки. Проверьте API ключ."
        logger.error(error_msg)
        return [error_msg]
    logger.debug("Клиент OpenAI инициализирован.")

    # --- Предварительная проверка запросов --- #
    valid_requests_with_indices = []
    for i, req_data in enumerate(image_requests):
        request_index = i + 1
        if not isinstance(req_data, dict):
            error_msg = f"Ошибка OpenAI запроса {request_index}: Элемент в 'image_requests' должен быть словарем."
            logger.error(error_msg)
            all_results.append(error_msg)
            error_requests_count += 1
            continue
        if not req_data.get("prompt"):
            error_msg = f"Ошибка OpenAI запроса {request_index}: Отсутствует обязательный ключ 'prompt'."
            logger.error(error_msg)
            all_results.append(error_msg)
            error_requests_count += 1
            continue
        # Доп. валидация reference_image_paths (проверка что это список строк, если есть)
        refs = req_data.get("reference_image_paths")
        if refs is not None and not (isinstance(refs, list) and all(isinstance(p, str) for p in refs)):
             error_msg = f"Ошибка OpenAI запроса {request_index}: 'reference_image_paths' должен быть списком строк."
             logger.error(error_msg)
             all_results.append(error_msg)
             error_requests_count += 1
             continue

        valid_requests_with_indices.append((req_data, request_index))

    if not valid_requests_with_indices:
        logger.warning("Нет валидных OpenAI запросов для обработки.")
        return all_results if all_results else ["Нет валидных OpenAI запросов для обработки."]

    # --- Создание базовой директории --- #
    try:
        save_directory = Path(base_output_dir)
        save_directory.mkdir(parents=True, exist_ok=True)
        logger.debug(f"Базовая директория для OpenAI изображений: {save_directory.resolve()}")
    except OSError as e:
        error_msg = f"Ошибка создания базовой директории {save_directory}: {e}"
        logger.exception(error_msg)
        all_results.insert(0, error_msg)
        return all_results
    except Exception as e:
        error_msg = f"Неожиданная ошибка при создании директории {base_output_dir}: {e}"
        logger.exception(error_msg)
        all_results.insert(0, error_msg)
        return all_results

    # --- Параллельная обработка --- #
    futures = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        logger.info(f"Отправка {len(valid_requests_with_indices)} валидных OpenAI запросов в ThreadPoolExecutor...")
        for req_data, req_index in valid_requests_with_indices:
            future = executor.submit(
                _process_single_openai_image_request,
                client, # Передаем КЛИЕНТ
                req_data,
                req_index,
                save_directory
            )
            futures.append(future)

        logger.info("Ожидание завершения обработки OpenAI запросов...")
        for future in concurrent.futures.as_completed(futures):
            try:
                result = future.result() # Список путей или строка ошибки
                if isinstance(result, list):
                    all_results.extend(result)
                    processed_files_count += len(result)
                elif isinstance(result, str):
                    all_results.append(result)
                    error_requests_count += 1
                else:
                    error_msg = f"Неожиданный тип результата из потока OpenAI: {type(result)}"
                    logger.error(error_msg)
                    all_results.append(error_msg)
                    error_requests_count += 1
            except Exception as e:
                error_msg = f"Критическая ошибка при получении результата из потока OpenAI: {e}"
                logger.exception(error_msg)
                all_results.append(error_msg)
                error_requests_count += 1

    logger.info(
        f"generate_and_save_multiple_openai_images (ПАРАЛЛЕЛЬНО) завершен. "
        f"Всего сохранено файлов: {processed_files_count}, запросов с ошибками: {error_requests_count}."
    )
    return all_results

# --- Убираем блок запуска сервера ---
# import asyncio # Больше не нужен
# from mcp.server.stdio import stdio_server # Больше не нужен

if __name__ == "__main__":
    logger.info("Запуск FastMCP сервера в режиме STDIN/STDOUT...")
    # Напрямую вызываем синхронный метод run(), который управляет циклом и stdio
    mcp.run()

    # Старый код удален:
    # async def main():
    #     logger.info("Запуск FastMCP сервера в режиме STDIN/STDOUT...")
    #     async with stdio_server() as (read_stream, write_stream):
    #         await mcp.run() # <-- Ошибка здесь
    #
    # asyncio.run(main())
