import os
import requests
import logging
from dotenv import load_dotenv
from pathlib import Path
import datetime

# Import the function from the existing client
try:
    from replicate_client import call_replicate_api
except ImportError:
    print("Ошибка: Не удалось импортировать 'call_replicate_api' из replicate_client.py")
    print("Убедитесь, что файл replicate_client.py находится в той же директории.")
    exit(1)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables (for REPLICATE_API_TOKEN)
load_dotenv()
if not os.getenv("REPLICATE_API_TOKEN"):
    logger.error("Токен REPLICATE_API_TOKEN не найден в .env файле.")
    exit(1)

# --- Parameters for Image Generation ---
generation_params = {
    "lora_trigger_word": "cat",
    "num_outputs": 1,
    "output_dir": "/Users/aliya/Новые видео", # Decoded from user input
    "aspect_ratio": "9:16",
    "lora_hf_id": "Aliyasta/kitten",
    "prompt": "A cat in Istanbul drinking coffee, cat"
}

def download_image(url: str, save_path: Path):
    """Downloads an image from a URL and saves it to the specified path."""
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()  # Raise an exception for bad status codes

        with open(save_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        logger.info(f"Изображение успешно сохранено: {save_path}")
        return True
    except requests.exceptions.RequestException as e:
        logger.error(f"Ошибка при скачивании изображения {url}: {e}")
        return False
    except IOError as e:
        logger.error(f"Ошибка при сохранении файла {save_path}: {e}")
        return False

def main():
    logger.info("Запуск теста генерации изображений...")

    output_dir = Path(generation_params["output_dir"])
    # Create the output directory if it doesn't exist
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Директория для сохранения создана/найдена: {output_dir}")
    except OSError as e:
        logger.error(f"Не удалось создать директорию {output_dir}: {e}")
        return # Exit if we can't create the directory

    # Call the Replicate API
    image_urls_or_error = call_replicate_api(
        prompt=generation_params["prompt"],
        num_outputs=generation_params["num_outputs"],
        aspect_ratio=generation_params["aspect_ratio"],
        lora_hf_id=generation_params["lora_hf_id"],
        lora_trigger_word=generation_params["lora_trigger_word"]
    )

    # Check for errors from the API call
    if isinstance(image_urls_or_error, str):
        logger.error(f"Ошибка при вызове Replicate API: {image_urls_or_error}")
        return

    if not image_urls_or_error:
        logger.warning("Replicate API не вернул URL изображений.")
        return

    logger.info(f"Получено {len(image_urls_or_error)} URL изображений от Replicate.")

    # Download and save the images
    download_count = 0
    for i, url in enumerate(image_urls_or_error):
        # Generate a unique filename
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"generated_image_{timestamp}_{i+1}.png" # Assuming PNG format from replicate_client
        save_path = output_dir / filename

        logger.info(f"Скачивание изображения {i+1}/{len(image_urls_or_error)} с URL: {url}")
        if download_image(url, save_path):
            download_count += 1

    logger.info(f"Завершено. Скачано {download_count} из {len(image_urls_or_error)} изображений в {output_dir}")

if __name__ == "__main__":
    main() 