import os
from openai import OpenAI, OpenAIError
from dotenv import load_dotenv
import logging
import base64
from typing import List, Optional # Добавляем для типизации

# Получаем логгер для этого модуля
logger = logging.getLogger(__name__)

# Загрузка API ключа из .env
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# --- Клиент OpenAI ---
# Можно инициализировать здесь один раз или передавать/создавать в функции
# Пока создадим функцию для инициализации
def get_openai_client() -> Optional[OpenAI]: # Используем Optional из typing
    """Инициализирует и возвращает клиент OpenAI или None при ошибке."""
    if not OPENAI_API_KEY or OPENAI_API_KEY == 'YOUR_API_KEY_HERE': # Добавил проверку на плейсхолдер
        logger.error("Ключ OpenAI API не найден или не задан в файле .env (OPENAI_API_KEY='sk-...')")
        return None
    try:
        client = OpenAI(api_key=OPENAI_API_KEY)
        logger.debug("Клиент OpenAI успешно инициализирован.")
        return client
    except Exception as e:
        logger.exception(f"Не удалось инициализировать клиента OpenAI: {e}")
        return None

# --- Функция вызова API генерации/редактирования --- (Обновленная)
def call_openai_image_api(
    client: OpenAI,
    prompt: str,
    # Параметры для generate и edit (не все применимы к обоим)
    n: int = 1,
    size: str = "1024x1024",
    quality: str = "medium",
    # --- Новый параметр для референсных изображений --- #
    reference_image_paths: Optional[List[str]] = None # Список путей
) -> list[str] | str:
    """
    Отправляет запрос на генерацию или редактирование изображений в OpenAI API (gpt-image-1).
    Если предоставлены reference_image_paths, используется режим редактирования.

    Args:
        client: Инициализированный клиент OpenAI.
        prompt: Текстовый промпт для генерации или редактирования.
        n: Количество изображений для генерации (игнорируется при редактировании).
        size: Размер изображения ('1024x1024', '1536x1024', '1024x1536', 'auto').
        quality: Качество ('low', 'medium', 'high', 'auto').
        reference_image_paths: (Опционально) Список абсолютных путей к файлам-референсам для режима редактирования.

    Returns:
        Список строк с base64-кодированными данными изображений (обычно 1 для edit)
        или строка с сообщением об ошибке.
    """
    # Валидация общих параметров
    valid_sizes = ["1024x1024", "1536x1024", "1024x1536", "auto"]
    if size not in valid_sizes:
        logger.warning(f"Некорректный размер '{size}'. Используется '1024x1024'. Допустимые: {valid_sizes}")
        size = "1024x1024"
    valid_qualities = ["low", "medium", "high", "auto"]
    if quality not in valid_qualities:
        logger.warning(f"Некорректное качество '{quality}'. Используется 'medium'. Допустимые: {valid_qualities}")
        quality = "medium"

    opened_files = []
    try:
        # --- Режим Редактирования (если есть референсы) ---
        if reference_image_paths:
            logger.info(f"Режим редактирования OpenAI: {len(reference_image_paths)} референсов.")
            logger.debug(
                f"Вызов images.edit: size={size}, quality={quality}, prompt='{prompt[:60]}...'"
            )

            # Открываем файлы референсов
            for path_str in reference_image_paths:
                try:
                    # Важно открывать в бинарном режиме 'rb'
                    f = open(path_str, "rb")
                    opened_files.append(f)
                    logger.debug(f"Открыт файл референса: {path_str}")
                except FileNotFoundError:
                    logger.error(f"Файл референса не найден: {path_str}")
                    return f"Ошибка: Файл референса не найден: {path_str}"
                except Exception as e:
                    logger.exception(f"Ошибка при открытии файла референса {path_str}")
                    return f"Ошибка открытия файла референса: {path_str}"

            if not opened_files:
                 # Эта ситуация не должна возникать из-за проверок выше, но на всякий случай
                 return "Ошибка: Не удалось открыть ни одного файла референса."

            response = client.images.edit(
                model="gpt-image-1",
                image=opened_files, # Передаем список открытых файлов
                prompt=prompt,
                size=size,
                quality=quality,
                n=1 # Edit всегда возвращает 1 результат
                # response_format='b64_json' # Ожидаем по умолчанию
            )
            logger.info("Вызов client.images.edit завершен.")

        # --- Режим Генерации (если нет референсов) ---
        else:
            logger.info("Режим генерации OpenAI.")
            logger.debug(
                f"Вызов images.generate: n={n}, size={size}, quality={quality}, prompt='{prompt[:60]}...'"
            )
            response = client.images.generate(
                model="gpt-image-1",
                prompt=prompt,
                n=n,
                size=size,
                quality=quality
                # response_format='b64_json' # Ожидаем по умолчанию
            )
            logger.info("Вызов client.images.generate завершен.")

        # --- Обработка ответа (одинакова для edit и generate) ---
        image_data_list = []
        if response.data:
            for image_obj in response.data:
                if image_obj.b64_json:
                    image_data_list.append(image_obj.b64_json)
                else:
                    logger.warning("Объект изображения в ответе API не содержит b64_json.")
        if not image_data_list:
            logger.warning(f"OpenAI API вернул ответ, но не содержит данных b64_json. Ответ: {response}")
            return "Ошибка: Ответ API не содержит ожидаемых данных b64_json."

        logger.info(f"OpenAI API успешно вернул {len(image_data_list)} изображений (base64).")
        return image_data_list

    except OpenAIError as e:
        error_message = f"Ошибка OpenAI API: {e}"
        logger.error(error_message)
        if hasattr(e, 'http_status'): logger.error(f"HTTP Status: {e.http_status}")
        if hasattr(e, 'code'): logger.error(f"Error Code: {e.code}")
        if hasattr(e, 'body') and isinstance(e.body, dict) and 'error' in e.body and isinstance(e.body['error'], dict) and 'message' in e.body['error']:
             error_message = f"Ошибка OpenAI API: {e.body['error']['message']}"
             logger.error(f"Подробное сообщение об ошибке: {e.body['error']['message']}")
        return error_message
    except Exception as e:
        logger.exception("Непредвиденная ошибка при вызове OpenAI API (генерация/редактирование)")
        return f"Непредвиденная ошибка при вызове OpenAI API: {e}"
    finally:
        # --- Гарантированное закрытие всех открытых файлов референсов ---
        closed_count = 0
        for f in opened_files:
            try:
                if not f.closed:
                    f.close()
                    closed_count += 1
            except Exception as close_err:
                # Логируем ошибку закрытия, но не прерываем основной процесс
                logger.error(f"Ошибка при закрытии файла референса {f.name} в блоке finally: {close_err}")
        if closed_count > 0:
            logger.debug(f"Закрыто {closed_count} файлов референсов в блоке finally.") 