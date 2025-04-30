import os
import replicate
from dotenv import load_dotenv
import logging
import sys

# --- Настройка логирования ---
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    handlers=[logging.StreamHandler(sys.stdout)])
logger = logging.getLogger(__name__)
logging.getLogger("replicate").setLevel(logging.INFO) # Уменьшим шум от библиотеки

# --- Загрузка ключа API ---
load_dotenv()
REPLICATE_API_KEY = os.getenv("REPLICATE_API_TOKEN")

if not REPLICATE_API_KEY:
    logger.error("Ошибка: Ключ REPLICATE_API_TOKEN не найден в .env или переменных окружения.")
    sys.exit(1)

# --- ID Модели (пример) ---
# Используем ту же модель, что и в сервере
MODEL_ID = "lucataco/flux-schnell-lora:2a6b576af31790b470f0a8442e1e9791213fa13799cbb65a9fc1436e96389574"

def test_timeout_initialization():
    """
    Проверяет инициализацию клиента Replicate с таймаутом
    и выполняет один вызов API.
    """
    logger.info("--- Начало теста инициализации с таймаутом ---")
    test_passed = False
    try:
        logger.debug("Инициализация клиента Replicate с timeout=600...")
        # Инициализируем клиент с указанным таймаутом
        client = replicate.Client(api_token=REPLICATE_API_KEY, timeout=600)
        logger.info("Клиент успешно инициализирован с timeout=600.")

        # Пробуем выполнить простой запрос
        input_params = {
            "prompt": "a cute cat",
            "aspect_ratio": "1:1",
            "num_outputs": 1,
        }
        logger.info(f"Выполнение тестового запроса к модели {MODEL_ID}...")
        output = client.run(MODEL_ID, input=input_params)

        # Проверяем базовый результат
        if isinstance(output, list) and len(output) > 0 and isinstance(output[0], str):
             logger.info(f"Тестовый запрос успешно выполнен. Получен URL: {output[0]}")
             test_passed = True
        # Новая проверка для FileOutput (как в replicate_client.py)
        elif hasattr(output, 'url') and isinstance(output.url, str):
            logger.info(f"Тестовый запрос успешно выполнен. Получен URL: {output.url}")
            test_passed = True
        # Еще один вариант для моделей типа flux-schnell, которые возвращают список FileOutput
        elif isinstance(output, list) and len(output) > 0 and hasattr(output[0], 'url') and isinstance(output[0].url, str):
            logger.info(f"Тестовый запрос успешно выполнен. Получен URL: {output[0].url}")
            test_passed = True
        else:
             logger.error(f"Тестовый запрос выполнен, но результат неожиданного типа: {type(output)}")

    except replicate.exceptions.ReplicateError as e:
        # Ловим ошибки Replicate (включая возможные таймауты от *этого* короткого запроса,
        # или ошибки 4xx/5xx)
        logger.error(f"Ошибка API Replicate во время теста: {e}")
    except Exception as e:
        # Ловим любые другие ошибки при инициализации или выполнении
        logger.exception(f"Непредвиденная ошибка во время теста: {e}")

    if test_passed:
        logger.info("--- Тест инициализации с таймаутом УСПЕШНО пройден ---")
    else:
        logger.error("--- Тест инициализации с таймаутом НЕ ПРОЙДЕН ---")

if __name__ == "__main__":
    test_timeout_initialization() 