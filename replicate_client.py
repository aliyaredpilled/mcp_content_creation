import os
import replicate
import traceback
from dotenv import load_dotenv
import logging

# Получаем логгер для этого модуля
logger = logging.getLogger(__name__)

# --- Константы и Настройки ---
MODEL_ID = "lucataco/flux-schnell-lora:2a6b576af31790b470f0a8442e1e9791213fa13799cbb65a9fc1436e96389574"
VIDEO_MODEL_ID = "minimax/video-01-live"
KLING_MODEL_ID = "kwaivgi/kling-v1.6-pro"

# Загрузка API ключа из .env
load_dotenv()
REPLICATE_API_KEY = os.getenv("REPLICATE_API_TOKEN")

if not REPLICATE_API_KEY:
    logger.warning("Ключ REPLICATE_API_TOKEN не найден. Запросы к Replicate API не будут работать.")

def call_replicate_api(
    prompt: str,
    num_outputs: int,
    aspect_ratio: str,
    lora_hf_id: str | None = None,
    lora_trigger_word: str | None = None,
) -> list[str] | str:
    """
    Отправляет запрос на генерацию изображений в Replicate API.

    Args:
        prompt: Текстовый промпт для генерации.
        num_outputs: Количество изображений для генерации.
        aspect_ratio: Соотношение сторон изображения.
        lora_hf_id: (Опционально) Идентификатор LoRA на Hugging Face.
        lora_trigger_word: (Опционально) Ключевое слово для активации LoRA.

    Returns:
        Список URL-адресов сгенерированных изображений или строка с сообщением об ошибке.
    """
    logger.debug(f"Вызов call_replicate_api с lora='{lora_hf_id}'")
    if not REPLICATE_API_KEY:
        logger.error("REPLICATE_API_TOKEN не настроен.")
        return "Ошибка: REPLICATE_API_TOKEN не настроен."

    if lora_hf_id and lora_trigger_word and lora_trigger_word not in prompt:
         logger.warning(f"Указан lora_hf_id, но lora_trigger_word '{lora_trigger_word}' не найден в промпте.")

    try:
        # Устанавливаем таймаут в 600 секунд (10 минут)
        client = replicate.Client(api_token=REPLICATE_API_KEY, timeout=600)

        input_params = {
            "prompt": prompt,
            "aspect_ratio": aspect_ratio,
            "num_outputs": num_outputs,
        }

        if lora_hf_id:
            input_params["hf_lora"] = lora_hf_id
            lora_info = f", LoRA: {lora_hf_id}"
        else:
            lora_info = ""

        logger.info(f"Запрос Replicate ({MODEL_ID.split(':')[0]}): N={num_outputs}, AR={aspect_ratio}{lora_info}, Prompt: '{prompt[:60]}...'" )

        output = client.run(MODEL_ID, input=input_params)

        if not isinstance(output, list):
            error_message = f"Replicate API вернул не список URL: {type(output)}"
            logger.error(error_message)
            return f"Ошибка: {error_message}"

        logger.info(f"Replicate API вернул {len(output)} URL.")
        return output

    except replicate.exceptions.ReplicateError as e:
        error_message = f"Ошибка API Replicate: {e}"
        logger.error(error_message)
        return error_message
    except Exception as e:
        logger.exception("Непредвиденная ошибка при вызове Replicate API изображений")
        # Возвращаем конкретное сообщение об ошибке
        return f"Непредвиденная ошибка при вызове Replicate API изображений: {e}"

def call_replicate_video_api(
    prompt: str,
    model_name: str,
    first_frame_image_path: str | None = None,
    end_image_path: str | None = None,
) -> str:
    """
    Отправляет запрос на генерацию видео в Replicate API,
    позволяя выбрать модель (minimax или kling) и опционально
    используя начальный и/или конечный кадр (конечный только для kling).

    Args:
        prompt: Текстовый промпт для генерации видео.
        model_name: Название модели для использования ('minimax' или 'kling'). Лучше клинг!)
        first_frame_image_path: (Опционально) Путь к файлу изображения для первого кадра.
        end_image_path: (Опционально) Путь к файлу изображения для последнего кадра (только для 'kling').

    Returns:
        URL сгенерированного видео или строка с сообщением об ошибке.
    """
    logger.debug(f"Вызов call_replicate_video_api (модель: {model_name}) с frame='{first_frame_image_path}', end_frame='{end_image_path}'")
    if not REPLICATE_API_KEY:
        logger.error("REPLICATE_API_TOKEN не настроен.")
        return "Ошибка: REPLICATE_API_TOKEN не настроен."

    # Проверяем корректность имени модели
    if model_name not in ["minimax", "kling"]:
        error_message = f"Неизвестное имя модели: '{model_name}'. Доступные: 'minimax', 'kling'."
        logger.error(error_message)
        return f"Ошибка: {error_message}"

    first_frame_image_content = None
    end_image_content = None
    first_image_info = "Нет"
    end_image_info = "Нет"

    # Обработка файла первого кадра
    if first_frame_image_path:
        if not os.path.exists(first_frame_image_path):
            error_message = f"Файл первого кадра не найден: {first_frame_image_path}"
            logger.error(error_message)
            # Важно закрыть файл конечного кадра, если он был открыт до этой ошибки
            if end_image_content and not end_image_content.closed: end_image_content.close()
            return f"Ошибка: {error_message}"
        try:
            logger.debug(f"Открытие файла первого кадра: {first_frame_image_path}")
            first_frame_image_content = open(first_frame_image_path, "rb")
            first_image_info = os.path.basename(first_frame_image_path) # Используем basename для краткости
        except Exception as e:
            logger.exception(f"Ошибка при открытии файла первого кадра {first_frame_image_path}")
            # Важно закрыть файл конечного кадра, если он был открыт до этой ошибки
            if end_image_content and not end_image_content.closed: end_image_content.close()
            return f"Ошибка при открытии файла первого кадра"

    # Обработка файла конечного кадра (только для kling)
    if model_name == "kling" and end_image_path:
        if not os.path.exists(end_image_path):
            error_message = f"Файл конечного кадра не найден: {end_image_path}"
            logger.error(error_message)
            # Важно закрыть файл первого кадра, если он был открыт до этой ошибки
            if first_frame_image_content and not first_frame_image_content.closed: first_frame_image_content.close()
            return f"Ошибка: {error_message}"
        try:
            logger.debug(f"Открытие файла конечного кадра: {end_image_path}")
            end_image_content = open(end_image_path, "rb")
            end_image_info = os.path.basename(end_image_path) # Используем basename для краткости
        except Exception as e:
            logger.exception(f"Ошибка при открытии файла конечного кадра {end_image_path}")
            # Важно закрыть файл первого кадра, если он был открыт до этой ошибки
            if first_frame_image_content and not first_frame_image_content.closed: first_frame_image_content.close()
            return f"Ошибка при открытии файла конечного кадра"
    elif model_name != "kling" and end_image_path:
         logger.warning(f"Параметр end_image_path указан, но будет проигнорирован для модели '{model_name}'.")


    try:
        # Устанавливаем таймаут в 600 секунд (10 минут)
        client = replicate.Client(api_token=REPLICATE_API_KEY, timeout=600)
        _model_id_to_use = None
        input_params = {"prompt": prompt}
        log_params_info = [] # Собираем инфо о параметрах для лога

        # Настройка параметров в зависимости от модели
        if model_name == "minimax":
            _model_id_to_use = VIDEO_MODEL_ID
            input_params["prompt_optimizer"] = True
            log_params_info.append("PromptOpt=True")
        elif model_name == "kling":
            _model_id_to_use = KLING_MODEL_ID
            input_params["aspect_ratio"] = "9:16"
            log_params_info.append("AR=9:16")
            if end_image_content:
                input_params["end_image"] = end_image_content
                log_params_info.append(f"EndFrame={end_image_info}")

        # Добавляем первый кадр, если есть
        if first_frame_image_content:
            # Используем правильное имя параметра для каждой модели
            param_name = "start_image" if model_name == "kling" else "first_frame_image"
            input_params[param_name] = first_frame_image_content
            log_params_info.append(f"Frame({param_name})={first_image_info}")
        else:
            # Если для Kling нет начального кадра, но есть конечный, это нормально
            if model_name != "kling" or not end_image_content:
                 # Для Minimax или если у Kling нет ни start ни end - добавляем "Frame=Нет"
                 log_params_info.append("Frame=Нет")
            # Если у Kling нет start_image, но есть end_image, мы его уже добавили в лог выше

        # Формируем строку параметров для лога
        params_log_str = ", ".join(log_params_info)
        logger.info(f"Запрос Replicate Video ({_model_id_to_use}): {params_log_str}, Prompt: '{prompt[:60]}...'" )

        # --- Вызов API ---
        # Файлы должны оставаться открытыми во время вызова run
        output_object = client.run(_model_id_to_use, input=input_params)

        # --- Новая проверка и извлечение URL ---
        # Проверяем, что это FileOutput или хотя бы имеет атрибут 'url'
        if hasattr(output_object, 'url'):
            video_url = output_object.url # Извлекаем URL
            # Можно добавить проверку, что video_url это строка, если нужно
            if not isinstance(video_url, str):
                 error_message = f"Атрибут 'url' объекта FileOutput не является строкой: {type(video_url)}"
                 logger.error(error_message)
                 return f"Ошибка: {error_message}"
        # Если это неожиданно строка (старое поведение?) - тоже принимаем
        elif isinstance(output_object, str):
             logger.warning("Replicate API вернул строку вместо объекта FileOutput. Используем строку как URL.")
             video_url = output_object
        # Если ни то, ни другое - ошибка
        else:
            error_message = f"Replicate API вернул неожиданный тип для видео: {type(output_object)}"
            logger.error(error_message)
            return f"Ошибка: {error_message}"

        logger.info(f"Replicate API вернул URL видео: {video_url}")
        return video_url # Возвращаем извлеченный URL

    except replicate.exceptions.ReplicateError as e:
        error_message = f"Ошибка API Replicate (видео): {e}"
        logger.error(error_message)
        return error_message
    except Exception as e:
        logger.exception("Непредвиденная ошибка при вызове Replicate API (видео)")
        return f"Непредвиденная ошибка при вызове Replicate API (видео): {e}"
    finally:
        # Гарантированное закрытие ОБОИХ файлов
        if first_frame_image_content and not first_frame_image_content.closed:
             try:
                 logger.debug(f"Закрытие файла первого кадра {first_frame_image_path} в блоке finally.")
                 first_frame_image_content.close()
             except Exception as close_err:
                 logger.error(f"Ошибка при закрытии файла первого кадра {first_frame_image_path} в finally: {close_err}")
        if end_image_content and not end_image_content.closed:
            try:
                logger.debug(f"Закрытие файла конечного кадра {end_image_path} в блоке finally.")
                end_image_content.close()
            except Exception as close_err:
                 logger.error(f"Ошибка при закрытии файла конечного кадра {end_image_path} в finally: {close_err}") 