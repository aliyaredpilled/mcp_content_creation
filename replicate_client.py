import os
import replicate
import traceback
from dotenv import load_dotenv

# --- Константы и Настройки ---
MODEL_ID = "lucataco/flux-schnell-lora:2a6b576af31790b470f0a8442e1e9791213fa13799cbb65a9fc1436e96389574"
VIDEO_MODEL_ID = "minimax/video-01-live"

# Загрузка API ключа из .env
load_dotenv()
REPLICATE_API_KEY = os.getenv("REPLICATE_API_TOKEN")

if not REPLICATE_API_KEY:
    print("Предупреждение: Ключ REPLICATE_API_TOKEN не найден. Запросы к Replicate API не будут работать.")

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
    if not REPLICATE_API_KEY:
        return "Ошибка: REPLICATE_API_TOKEN не настроен."

    if lora_hf_id and lora_trigger_word and lora_trigger_word not in prompt:
         print(f"Предупреждение: Указан lora_hf_id, но lora_trigger_word '{lora_trigger_word}' не найден в промпте.")

    try:
        client = replicate.Client(api_token=REPLICATE_API_KEY)

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

        print(f"Запрос Replicate ({MODEL_ID.split(':')[0]}): N={num_outputs}, AR={aspect_ratio}{lora_info}, Prompt: '{prompt[:60]}...'" )

        output = client.run(MODEL_ID, input=input_params)

        if not isinstance(output, list):
            error_message = f"Ошибка: Replicate API вернул не список URL: {type(output)}"
            print(error_message)
            return error_message

        print(f"Replicate вернул {len(output)} URL.")
        return output

    except replicate.exceptions.ReplicateError as e:
        error_message = f"Ошибка API Replicate: {e}"
        print(error_message)
        return error_message
    except Exception as e:
        error_message = f"Непредвиденная ошибка при вызове Replicate API: {e}"
        print(error_message)
        traceback.print_exc()
        return error_message

def call_replicate_video_api(
    prompt: str,
    first_frame_image_path: str = None,
) -> str | str:
    """
    Отправляет запрос на генерацию видео в Replicate API,
    опционально используя начальный кадр.

    Args:
        prompt: Текстовый промпт для генерации видео.
        first_frame_image_path: (Опционально) Путь к файлу изображения для первого кадра.

    Returns:
        URL сгенерированного видео или строка с сообщением об ошибке.
    """
    if not REPLICATE_API_KEY:
        return "Ошибка: REPLICATE_API_TOKEN не настроен."

    image_file_content = None
    image_info = "Нет"
    if first_frame_image_path:
        if not os.path.exists(first_frame_image_path):
            error_message = f"Ошибка: Файл первого кадра не найден: {first_frame_image_path}"
            print(error_message)
            return error_message
        try:
            image_file_content = open(first_frame_image_path, "rb")
            image_info = first_frame_image_path
        except Exception as e:
            error_message = f"Ошибка при открытии файла {first_frame_image_path}: {e}"
            print(error_message)
            return error_message

    try:
        client = replicate.Client(api_token=REPLICATE_API_KEY)

        input_params = {
            "prompt": prompt,
            "prompt_optimizer": True,
        }

        if image_file_content:
            input_params["first_frame_image"] = image_file_content

        print(f"Запрос Replicate Video ({VIDEO_MODEL_ID}): Кадр={image_info}, PromptOpt=True, Prompt: '{prompt[:60]}...'" )

        if image_file_content:
            with image_file_content:
                output_url = client.run(VIDEO_MODEL_ID, input=input_params)
        else:
            output_url = client.run(VIDEO_MODEL_ID, input=input_params)

        if not isinstance(output_url, str):
            error_message = f"Ошибка: Replicate API вернул не URL для видео: {type(output_url)}"
            print(error_message)
            if isinstance(output_url, list) and len(output_url) == 1 and isinstance(output_url[0], str):
                print("Предполагаем, что первый элемент списка - это URL.")
                output_url = output_url[0]
            else:
                return error_message

        print(f"Replicate вернул URL видео: {output_url}")
        return output_url

    except replicate.exceptions.ReplicateError as e:
        error_message = f"Ошибка API Replicate (видео): {e}"
        print(error_message)
        return error_message
    except Exception as e:
        error_message = f"Непредвиденная ошибка при вызове Replicate API (видео): {e}"
        print(error_message)
        traceback.print_exc()
        return error_message
    finally:
        if image_file_content and not image_file_content.closed:
             try:
                 image_file_content.close()
             except Exception as close_err:
                 print(f"Ошибка при закрытии файла {first_frame_image_path} в finally: {close_err}") 