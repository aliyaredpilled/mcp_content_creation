import os
import traceback
from dotenv import load_dotenv
from elevenlabs.client import ElevenLabs
from elevenlabs.core import ApiError

# Загрузка API ключа из .env
load_dotenv()
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")

# --- Константы по умолчанию --- #
DEFAULT_VOICE_ID = "43h7ymOnaaYdWr3dRbsS" # Новый ID по умолчанию
DEFAULT_MODEL_ID = "eleven_multilingual_v2"
DEFAULT_OUTPUT_FORMAT = "mp3_44100_128"
# ----------------------------- #

if not ELEVENLABS_API_KEY:
    print("Предупреждение: Ключ ELEVENLABS_API_KEY не найден. Запросы к ElevenLabs API не будут работать.")

def call_elevenlabs_tts_api(
    text: str,
    voice_id: str = None, # Передаем None чтобы использовать default в клиенте
    model_id: str = None, # Передаем None чтобы использовать default в клиенте
    output_format: str = DEFAULT_OUTPUT_FORMAT,
) -> bytes | str:
    """
    Отправляет запрос на генерацию речи в ElevenLabs TTS API.

    Args:
        text: Текст для озвучки.
        voice_id: Идентификатор голоса ElevenLabs (по умолчанию используется в клиенте).
        model_id: Идентификатор модели ElevenLabs (по умолчанию используется в клиенте).
        output_format: Формат вывода аудио (по умолчанию: {DEFAULT_OUTPUT_FORMAT}).

    Returns:
        Байты сгенерированного аудиофайла или строка с сообщением об ошибке.
    """
    if not ELEVENLABS_API_KEY:
        return "Ошибка: ELEVENLABS_API_KEY не настроен."

    # Используем значения по умолчанию из констант, если аргументы None
    actual_voice_id = voice_id if voice_id else DEFAULT_VOICE_ID
    actual_model_id = model_id if model_id else DEFAULT_MODEL_ID

    try:
        # print(f"Инициализация клиента ElevenLabs...") # Убрал
        client = ElevenLabs(api_key=ELEVENLABS_API_KEY)

        print(f"Запрос ElevenLabs TTS: Voice={actual_voice_id}, Model={actual_model_id}, Format={output_format}, Text: '{text[:60]}...'")

        # Вызов API для генерации аудио
        audio_result = client.text_to_speech.convert(
            text=text,
            voice_id=actual_voice_id, # Передаем актуальное значение
            model_id=actual_model_id, # Передаем актуальное значение
            output_format=output_format,
        )

        # --- Обработка результата (может быть bytes или generator) ---
        audio_bytes = b""
        if isinstance(audio_result, bytes):
            audio_bytes = audio_result
        elif hasattr(audio_result, '__iter__') and not isinstance(audio_result, (str, bytes)):
            # print("API вернул генератор, собираем байты...") # Убрал
            for chunk in audio_result:
                if isinstance(chunk, bytes):
                    audio_bytes += chunk
                else:
                    error_message = f"Ошибка: Генератор ElevenLabs содержит не байты, а {type(chunk)}"
                    print(error_message)
                    return error_message
            if not audio_bytes:
                 error_message = "Ошибка: Генератор ElevenLabs не вернул байтов."
                 print(error_message)
                 return error_message
        else:
            error_message = f"Ошибка: ElevenLabs API вернул неожиданный тип: {type(audio_result)}"
            print(error_message)
            return error_message
        # --------------------------------------------------------------

        print(f"ElevenLabs API успешно вернул {len(audio_bytes)} байт аудио.")
        return audio_bytes

    except ApiError as e:
        error_message = f"Ошибка API ElevenLabs: {e.status_code} - {e.message}"
        print(error_message)
        return error_message
    except Exception as e:
        error_message = f"Непредвиденная ошибка при вызове API ElevenLabs: {e}"
        print(error_message)
        traceback.print_exc()
        return error_message 