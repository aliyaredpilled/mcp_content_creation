import os
import json
from pathlib import Path
import logging # Для создания простого логгера
import sys
from replicate_mcp_server import create_capcut_project

def run_manual_test(test_data: dict, capcut_template_path: str, capcut_projects_root: str):
    """
    Запускает ручной тест для создания проекта CapCut.
    """
    
    # Настройка простого логгера для этого теста
    test_logger = logging.getLogger("manual_capcut_test")
    if not test_logger.hasHandlers():
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        test_logger.addHandler(handler)
        test_logger.setLevel(logging.DEBUG) # Устанавливаем DEBUG, чтобы видеть все логи от _generate_capcut_project_logic

    test_logger.info(f"--- Запуск ручного теста для проекта: {test_data.get('project_name')} ---")

    # Установка переменных окружения, если они еще не установлены
    # (create_capcut_project будет их читать)
    os.environ["CAPCUT_TEMPLATE_PATH"] = capcut_template_path
    os.environ["CAPCUT_PROJECTS_ROOT"] = capcut_projects_root
    
    project_name = test_data.get("project_name")
    media_items_info = test_data.get("media_items_info")
    orientation = test_data.get("orientation")
    # Можно добавить project_canvas_width/height, если они есть в test_data
    project_canvas_width = test_data.get("project_canvas_width")
    project_canvas_height = test_data.get("project_canvas_height")


    if not project_name or not media_items_info:
        test_logger.error("Ошибка в тестовых данных: отсутствует project_name или media_items_info.")
        return

    # Проверка существования медиафайлов
    all_media_exist = True
    for i, item in enumerate(media_items_info):
        item_path_str = item.get('path')
        if not item_path_str or not Path(item_path_str).is_file():
            test_logger.error(f"Медиафайл не найден: '{item_path_str}' (элемент {i})")
            all_media_exist = False
    
    if not all_media_exist:
        test_logger.error("Один или несколько медиафайлов не найдены. Тест прерван.")
        return

    # Вызов функции create_capcut_project (которая является оберткой для MCP)
    # Передаем наш test_logger
    result = create_capcut_project(
        project_name=project_name,
        media_items_info=media_items_info,
        orientation=orientation,
        project_canvas_width=project_canvas_width, # Передаем, если есть
        project_canvas_height=project_canvas_height, # Передаем, если есть
        logger=test_logger 
    )

    test_logger.info(f"Результат вызова create_capcut_project: {result}")
    test_logger.info(f"--- Ручной тест для проекта '{project_name}' завершен ---")


if __name__ == "__main__":
    # --- Конфигурация для ручного теста ---
    # УКАЖИ ПРАВИЛЬНЫЕ ПУТИ ЗДЕСЬ!
    CAPCUT_TEMPLATE_PROJECT_PATH = "/Users/aliya/json2video-mcp-server/0510" 
    CAPCUT_PROJECTS_OUTPUT_ROOT = "/Users/aliya/Movies/CapCut/User Data/Projects/com.lveditor.draft"

    # Твои тестовые данные (как JSON, но здесь как Python словарь)
    test_project_data = {
      "project_name": "Колокольчик_Ручной_Тест_Вертикальный", # Изменим имя, чтобы не конфликтовать
      "media_items_info": [
        {
          "path": "/Users/aliya/json2video-mcp-server/Контент агент/semki_razdora_videos_no_end_frame/video_01.mp4",
          "type": "video", "timeline_start_ms": 0, "timeline_duration_ms": 5000,
          "source_start_ms": 0, "track_index": 0
        },
        {
          "path": "/Users/aliya/json2video-mcp-server/Контент агент/semki_razdora_videos_no_end_frame/video_02.mp4",
          "type": "video", "timeline_start_ms": 5000, "timeline_duration_ms": 5000,
          "source_start_ms": 0, "track_index": 0
        },
        # ... (остальные видеофайлы из твоего примера) ...
        {
          "path": "/Users/aliya/json2video-mcp-server/Контент агент/semki_razdora_videos_no_end_frame/video_03.mp4",
          "type": "video", "timeline_start_ms": 10000, "timeline_duration_ms": 5000,
          "source_start_ms": 0, "track_index": 0
        },
        {
          "path": "/Users/aliya/json2video-mcp-server/Контент агент/semki_razdora_videos_no_end_frame/video_04.mp4",
          "type": "video", "timeline_start_ms": 15000, "timeline_duration_ms": 5000,
          "source_start_ms": 0, "track_index": 0
        },
        {
          "path": "/Users/aliya/json2video-mcp-server/Контент агент/semki_razdora_videos_no_end_frame/video_05.mp4",
          "type": "video", "timeline_start_ms": 20000, "timeline_duration_ms": 5000,
          "source_start_ms": 0, "track_index": 0
        },
        {
          "path": "/Users/aliya/json2video-mcp-server/Контент агент/semki_razdora_videos_no_end_frame/video_06.mp4",
          "type": "video", "timeline_start_ms": 25000, "timeline_duration_ms": 5000,
          "source_start_ms": 0, "track_index": 0
        },
        {
          "path": "/Users/aliya/json2video-mcp-server/Контент агент/semki_razdora_videos_no_end_frame/video_07.mp4",
          "type": "video", "timeline_start_ms": 30000, "timeline_duration_ms": 5000,
          "source_start_ms": 0, "track_index": 0
        },
        {
          "path": "/Users/aliya/json2video-mcp-server/Контент агент/semki_razdora_videos_no_end_frame/video_08.mp4",
          "type": "video", "timeline_start_ms": 35000, "timeline_duration_ms": 5000,
          "source_start_ms": 0, "track_index": 0
        },
        {
          "path": "/Users/aliya/json2video-mcp-server/Контент агент/semki_razdora_videos_no_end_frame/video_09.mp4",
          "type": "video", "timeline_start_ms": 40000, "timeline_duration_ms": 5000,
          "source_start_ms": 0, "track_index": 0
        },
        {
          "path": "/Users/aliya/json2video-mcp-server/Контент агент/semki_razdora_audio/audio_01_ff0239.mp3",
          "type": "audio", "timeline_start_ms": 0, "timeline_duration_ms": 5694,
          "source_start_ms": 0, "track_index": 1 # Аудио на другой дорожке
        },
        # ... (остальные аудиофайлы из твоего примера) ...
        {
          "path": "/Users/aliya/json2video-mcp-server/Контент агент/semki_razdora_audio/audio_02_bd2cc6.mp3",
          "type": "audio", "timeline_start_ms": 5694, "timeline_duration_ms": 5381,
          "source_start_ms": 0, "track_index": 1
        },
        {
          "path": "/Users/aliya/json2video-mcp-server/Контент агент/semki_razdora_audio/audio_03_c0c47f.mp3",
          "type": "audio", "timeline_start_ms": 11075, "timeline_duration_ms": 5878,
          "source_start_ms": 0, "track_index": 1
        },
        {
          "path": "/Users/aliya/json2video-mcp-server/Контент агент/semki_razdora_audio/audio_04_a8e6f4.mp3",
          "type": "audio", "timeline_start_ms": 16953, "timeline_duration_ms": 8620,
          "source_start_ms": 0, "track_index": 1
        },
        {
          "path": "/Users/aliya/json2video-mcp-server/Контент агент/semki_razdora_audio/audio_05_f5d849.mp3",
          "type": "audio", "timeline_start_ms": 25573, "timeline_duration_ms": 7601,
          "source_start_ms": 0, "track_index": 1
        },
        {
          "path": "/Users/aliya/json2video-mcp-server/Контент агент/semki_razdora_audio/audio_06_43e12c.mp3",
          "type": "audio", "timeline_start_ms": 33174, "timeline_duration_ms": 5616,
          "source_start_ms": 0, "track_index": 1
        },
        {
          "path": "/Users/aliya/json2video-mcp-server/Контент агент/semki_razdora_audio/audio_07_1ecf46.mp3",
          "type": "audio", "timeline_start_ms": 38790, "timeline_duration_ms": 6217,
          "source_start_ms": 0, "track_index": 1
        },
        {
          "path": "/Users/aliya/json2video-mcp-server/Контент агент/semki_razdora_audio/audio_08_bbfcc1.mp3",
          "type": "audio", "timeline_start_ms": 45007, "timeline_duration_ms": 6948,
          "source_start_ms": 0, "track_index": 1
        },
        {
          "path": "/Users/aliya/json2video-mcp-server/Контент агент/semki_razdora_audio/audio_09_271e85.mp3",
          "type": "audio", "timeline_start_ms": 51955, "timeline_duration_ms": 8672,
          "source_start_ms": 0, "track_index": 1
        }
      ],
      "orientation": "vertical"
    }

    # Запуск теста
    run_manual_test(test_project_data, CAPCUT_TEMPLATE_PROJECT_PATH, CAPCUT_PROJECTS_OUTPUT_ROOT)

    # Можно добавить еще один тест с другими параметрами, например, горизонтальной ориентацией
    test_project_data_horizontal = {
      "project_name": "Колокольчик_Ручной_Тест_Горизонтальный",
      "media_items_info": [ # Возьмем только первые несколько для краткости
        {
          "path": "/Users/aliya/json2video-mcp-server/Контент агент/semki_razdora_videos_no_end_frame/video_01.mp4",
          "type": "video", "timeline_start_ms": 0, "timeline_duration_ms": 3000,
          "source_start_ms": 0, "track_index": 0
        },
        {
          "path": "/Users/aliya/json2video-mcp-server/Контент агент/semki_razdora_audio/audio_01_ff0239.mp3",
          "type": "audio", "timeline_start_ms": 0, "timeline_duration_ms": 3000,
          "source_start_ms": 0, "track_index": 1
        }
      ],
      "orientation": "horizontal"
    }
    run_manual_test(test_project_data_horizontal, CAPCUT_TEMPLATE_PROJECT_PATH, CAPCUT_PROJECTS_OUTPUT_ROOT)