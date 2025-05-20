import os
import fal_client
import requests
from pathlib import Path
import uuid
import subprocess
import json # <-- Add json import
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP
# Возвращаем импорты инструментов
from elevenlabs_client import call_elevenlabs_tts_api, DEFAULT_OUTPUT_FORMAT # <--- Импорт новой функции и константы
from openai_client import get_openai_client, call_openai_image_api # <--- Импорт новых функций OpenAI
from capcut_tool import _generate_capcut_project_logic # <-- Add import for capcut tool
import traceback # Добавляем импорт traceback сюда, т.к. он используется в блоке except
import logging
import sys # Для вывода в stderr
import concurrent.futures # <--- Добавляем импорт
import base64 # <--- Добавляем импорт для OpenAI
from typing import Optional, List, Dict
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
logging.getLogger("fal_client").setLevel(logging.WARNING)
logging.getLogger("elevenlabs").setLevel(logging.WARNING)
# ------------------------------------

# --- Константы и Настройки ---
DEFAULT_ASPECT_RATIO = "9:16" # Соотношение по умолчанию, если не указано

# Загрузка API ключа из .env
load_dotenv()

# --- Инициализация MCP Сервера ---
mcp = FastMCP("Fal Kling Video Generator")

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
        logger.error(f"Ошибка скачивания {url}: {e}", exc_info=True)
        return False
    except Exception as e:
        logger.exception(f"Неожиданная ошибка сохранения файла {save_path}")
        return False

def _try_open_file(file_path: Path):
    """Пытается открыть файл системной командой."""
    try:
        logger.info(f"Попытка открыть: {file_path}")
        subprocess.run(["open", str(file_path)], check=True, capture_output=True, text=True)
        logger.info(f"Команда 'open' для {file_path} выполнена.")
        return True
    except FileNotFoundError:
        logger.warning("Ошибка: Команда 'open' не найдена. Не могу открыть файл автоматически (только macOS).")
    except subprocess.CalledProcessError as open_err:
        error_details = open_err.stderr if open_err.stderr else str(open_err)
        logger.error(f"Ошибка при выполнении команды 'open' для {file_path}: {error_details}")
    except Exception as general_err:
        logger.exception(f"Неожиданная ошибка при открытии файла {file_path}")
    return False

# --- Новый синхронный инструмент для генерации видео через fal.ai Kling ---
def call_fal_kling_video_api(prompt, image_path_or_url, output_dir, filename_prefix="video", duration="5", aspect_ratio="9:16", negative_prompt="blur, distort, and low quality", cfg_scale=0.5):
    """
    Генерирует видео через fal.ai Kling 1.6 Image-to-Video.
    Если image_path_or_url — локальный путь, сначала загружает файл на fal.ai.
    Возвращает путь к сохранённому видео или строку ошибки.
    """
    try:
        # Определяем, локальный ли это файл
        if os.path.isfile(image_path_or_url):
            logger.info(f"Загружаем локальный файл на fal.ai: {image_path_or_url}")
            image_url = fal_client.upload_file(image_path_or_url)
            logger.info(f"Файл загружен, URL: {image_url}")
        else:
            image_url = image_path_or_url
            logger.info(f"Используем image_url: {image_url}")

        result = fal_client.subscribe(
            "fal-ai/kling-video/v1.6/standard/image-to-video",
            arguments={
                "prompt": prompt,
                "image_url": image_url,
                "duration": duration,
                "aspect_ratio": aspect_ratio,
                "negative_prompt": negative_prompt,
                "cfg_scale": cfg_scale,
            },
            with_logs=True,
            on_queue_update=lambda update: (
                [logger.info(f"fal.ai: {log['message']}") for log in getattr(update, 'logs', [])]
                if hasattr(update, 'logs') else None
            )
        )
        video_url = result["video"]["url"]
        logger.info(f"fal.ai вернул видео: {video_url}")

        # Сохраняем видео
        save_directory = Path(output_dir)
        save_directory.mkdir(parents=True, exist_ok=True)
        filename = f"{filename_prefix}_{uuid.uuid4().hex[:6]}.mp4"
        save_path = save_directory / filename
        logger.debug(f"Скачивание видео {video_url} в {save_path}...")
        if download_file(video_url, save_path):
            _try_open_file(save_path)
            return str(save_path)
        else:
            return f"Ошибка: Не удалось скачать видео по ссылке {video_url}"
    except Exception as e:
        logger.exception("Ошибка при генерации видео через fal.ai")
        return f"Ошибка при генерации видео через fal.ai: {e}"

# --- Инструмент MCP для Видео через fal.ai Kling ---
@mcp.tool()
def generate_and_save_video(
    prompt: str,
    output_dir: str,
    image_path_or_url: str,
    filename_prefix: str = "video",
    duration: str = "5",
    aspect_ratio: str = DEFAULT_ASPECT_RATIO,
    negative_prompt: str = "blur, distort, and low quality",
    cfg_scale: float = 0.5
) -> list[str]:
    """
    Генерирует видео с помощью fal.ai Kling 1.6 Image-to-Video,
    используя prompt и картинку (локальный путь или URL).
    Сохраняет видео в указанную папку и пытается открыть.
    Возвращает список с путем к сохраненному видеофайлу или сообщение об ошибке.
    """
    logger.info(
        f"Вызов generate_and_save_video (fal.ai Kling): "
        f"prompt='{prompt[:50]}...', dir='{output_dir}', image='{image_path_or_url}', prefix='{filename_prefix}', duration='{duration}', ar='{aspect_ratio}'"
    )
    try:
        result = call_fal_kling_video_api(
            prompt=prompt,
            image_path_or_url=image_path_or_url,
            output_dir=output_dir,
            filename_prefix=filename_prefix,
            duration=duration,
            aspect_ratio=aspect_ratio,
            negative_prompt=negative_prompt,
            cfg_scale=cfg_scale
        )
        return [result]
    except Exception as e:
        logger.exception("Ошибка при запуске генерации видео через fal.ai")
        return [f"Ошибка при запуске генерации видео через fal.ai: {e}"]

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


@mcp.tool() # Раскомментируй для использования в MCP
def create_capcut_project(
    project_name: str,
    media_items_info: List[Dict], 
    default_fps: float = 30.0,
    project_canvas_width: Optional[int] = None,
    project_canvas_height: Optional[int] = None,
    orientation: Optional[str] = None, # <--- ДОБАВЛЕН ПАРАМЕТР
    logger = None # Принимаем логгер, как и раньше
) -> List[str]:
    """
    Создает проект CapCut на основе предоставленных медиафайлов и информации о таймлайне.
    Пути к шаблону проекта и корневой папке проектов CapCut берутся из переменных окружения:
    CAPCUT_TEMPLATE_PATH и CAPCUT_PROJECTS_ROOT.

    Args:
        project_name: Имя нового проекта CapCut.
        media_items_info: Список словарей. Каждый словарь описывает медиафайл
                          для таймлайна и должен содержать:
                          - 'path' (str): Путь к медиафайлу.
                          - 'type' (str): 'video' или 'audio'.
                          - 'timeline_start_ms' (int): Время начала на таймлайне (в мс).
                          - 'timeline_duration_ms' (int): Длительность на таймлайне (в мс).
                          - 'source_start_ms' (int): Время начала внутри исходного файла (в мс).
                          - 'track_index' (int): Индекс дорожки (0, 1, ...).
        default_fps: (Опционально) FPS проекта (по умолчанию 30.0).
        project_canvas_width: (Опционально) Ширина холста проекта. 
                              Игнорируется, если задан 'orientation'.
        project_canvas_height: (Опционально) Высота холста проекта. 
                               Игнорируется, если задан 'orientation'.
        orientation: (Опционально) Ориентация холста ("vertical", "horizontal", "square"). 
                     Имеет приоритет над project_canvas_width/height.

    Returns:
        Список строк: Сообщение об успехе с путем к проекту или сообщение об ошибке.
    """
    # Настройка логгера, если не передан
    if logger is None:
        import logging as pylogging 
        import sys
        logger = pylogging.getLogger("create_capcut_project_tool_mcp_local")
        if not logger.hasHandlers():
            handler = pylogging.StreamHandler(sys.stdout)
            formatter = pylogging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(pylogging.DEBUG)

    base_template_project_path_str = os.environ.get("CAPCUT_TEMPLATE_PATH")
    output_project_root_path_str = os.environ.get("CAPCUT_PROJECTS_ROOT")

    logger.info(
        f"Вызов create_capcut_project: project_name='{project_name}', "
        f"media_items={len(media_items_info)}, fps={default_fps}, "
        f"canvas_w={project_canvas_width}, canvas_h={project_canvas_height}, orientation='{orientation}'. " # Добавлен orientation в лог
        f"ENV_TEMPLATE_PATH='{base_template_project_path_str}', ENV_PROJECTS_ROOT='{output_project_root_path_str}'"
    )

    # --- Валидация (остается такой же, как в твоем коде) ---
    if not project_name:
        logger.error("Ошибка валидации: 'project_name' не может быть пустым.")
        return ["Ошибка: 'project_name' не может быть пустым."]
    if not isinstance(media_items_info, list): # Проверка на тип list
        logger.error("Ошибка валидации: 'media_items_info' должен быть списком.")
        return ["Ошибка: 'media_items_info' должен быть списком."]
    # Если media_items_info пустой, это может быть допустимо (создание пустого проекта)
    # if not media_items_info: 
    #     logger.warning("Предупреждение: 'media_items_info' пуст. Проект будет создан без медиа на таймлайне.")
    
    if not base_template_project_path_str:
        logger.error("Ошибка валидации: Переменная окружения 'CAPCUT_TEMPLATE_PATH' не установлена или пуста.")
        return ["Ошибка: Переменная окружения 'CAPCUT_TEMPLATE_PATH' не установлена или пуста."]
    base_template_project_path = Path(base_template_project_path_str)
    if not base_template_project_path.is_dir():
        logger.error(f"Ошибка валидации: Путь к шаблону '{base_template_project_path_str}' не существует или не является директорией.")
        return [f"Ошибка: Путь к шаблону CapCut из ENV 'CAPCUT_TEMPLATE_PATH' ('{base_template_project_path_str}') не существует или не является директорией."]

    if not output_project_root_path_str:
        logger.error("Ошибка валидации: Переменная окружения 'CAPCUT_PROJECTS_ROOT' не установлена или пуста.")
        return ["Ошибка: Переменная окружения 'CAPCUT_PROJECTS_ROOT' не установлена или пуста."]
    output_project_root_path = Path(output_project_root_path_str)
    if not output_project_root_path.is_dir():
        try:
            output_project_root_path.mkdir(parents=True, exist_ok=True)
            logger.info(f"Создана корневая директория для проектов CapCut: '{output_project_root_path}'")
        except Exception as e:
            logger.error(f"Ошибка: Не удалось создать корневую директорию для проектов CapCut '{output_project_root_path_str}': {e}")
            return [f"Ошибка: Не удалось создать корневую директорию для проектов CapCut '{output_project_root_path_str}': {e}"]

    for i, item in enumerate(media_items_info):
        if not isinstance(item, dict):
            logger.error(f"Ошибка валидации: Элемент {i} в 'media_items_info' не является словарем.")
            return [f"Ошибка: Элемент {i} в 'media_items_info' не является словарем."]
        required_keys = ['path', 'type', 'timeline_start_ms', 'timeline_duration_ms', 'source_start_ms', 'track_index']
        missing_keys = [key for key in required_keys if key not in item]
        if missing_keys:
            logger.error(f"Ошибка валидации: Элемент {i} в 'media_items_info' не содержит обязательные ключи: {', '.join(missing_keys)}.")
            return [f"Ошибка: Элемент {i} в 'media_items_info' не содержит обязательные ключи: {', '.join(missing_keys)}."]
        
        item_path_str = item.get('path')
        if not item_path_str or not Path(item_path_str).is_file(): # Проверяем, что путь есть и это файл
            logger.error(f"Ошибка валидации: Файл '{item_path_str}' в 'media_items_info' (элемент {i}) не найден или путь не указан.")
            return [f"Ошибка: Файл '{item_path_str}' в 'media_items_info' (элемент {i}) не найден или путь не указан."]
        
        for key_to_check in ['timeline_start_ms', 'timeline_duration_ms', 'source_start_ms', 'track_index']:
            if not isinstance(item.get(key_to_check), int):
                logger.error(f"Ошибка валидации: Ключ '{key_to_check}' в элементе {i} 'media_items_info' должен быть int. Получено: {item.get(key_to_check)} (тип: {type(item.get(key_to_check))}).")
                return [f"Ошибка: Ключ '{key_to_check}' в элементе {i} 'media_items_info' должен быть целым числом (int). Получено: {item.get(key_to_check)} (тип: {type(item.get(key_to_check))})."]
        if item.get('type') not in ['video', 'audio']:
            logger.error(f"Ошибка валидации: Ключ 'type' в элементе {i} 'media_items_info' должен быть 'video' или 'audio'. Получено: {item.get('type')}.")
            return [f"Ошибка: Ключ 'type' в элементе {i} 'media_items_info' должен быть 'video' или 'audio'. Получено: {item.get('type')}."]

    try:
        logger.info(f"Вызов _generate_capcut_project_logic для '{project_name}' с ориентацией: '{orientation}'")
        
        # Прямой вызов _generate_capcut_project_logic с передачей всех параметров
        success = _generate_capcut_project_logic(
            project_name=project_name,
            media_items_info=media_items_info,
            base_template_project_path=base_template_project_path, # Передаем Path объект
            output_project_root_path=output_project_root_path, # Передаем Path объект
            default_fps=default_fps,
            project_canvas_width=project_canvas_width,
            project_canvas_height=project_canvas_height,
            orientation=orientation, # <--- ПЕРЕДАЕМ ORIENTATION
            logger_instance=logger 
        )

        if success:
            sanitized_name = project_name.replace(" ", "_").replace("/", "_").replace("\\", "_")
            project_folder_path = output_project_root_path / sanitized_name
            success_message = f"Проект CapCut '{project_name}' успешно создан. Путь: {project_folder_path.resolve()}"
            logger.info(success_message)
            return [success_message]
        else:
            error_message = f"Ошибка при генерации проекта CapCut '{project_name}' (_generate_capcut_project_logic вернул False)."
            logger.error(error_message)
            return [error_message]
            
    except ValueError as ve:
        logger.error(f"Ошибка входных данных при вызове _generate_capcut_project_logic: {ve}", exc_info=True)
        return [f"Ошибка входных данных: {ve}"]
    except Exception as e:
        # Используем traceback для получения полного стека ошибки, если он нужен
        # error_details = traceback.format_exc() 
        logger.error(f"Непредвиденная ошибка при создании проекта CapCut '{project_name}': {e}", exc_info=True)
        return [f"Непредвиденная ошибка при создании проекта: {e}"]

if __name__ == "__main__":
    logger.info("Запуск FastMCP сервера в режиме STDIN/STDOUT...")
    mcp.run()