import json
import os
import uuid
import subprocess
import time
import shutil
from pathlib import Path 
from typing import List, Dict, Optional, Union
import sys # Для print в stderr, если понадобится для локальной отладки get_media_metadata

# --- Конфигурация ---
FFPROBE_PATH = "ffprobe" # Убедись, что ffprobe доступен в PATH или укажи полный путь

def get_media_metadata(media_path: Union[str, Path], logger_instance=None) -> Optional[Dict]:
    """Получает метаданные для медиафайла (видео или аудио)."""
    def log_warning(message): # Внутренняя функция для логирования предупреждений
        if logger_instance: logger_instance.warning(message)
        # else: print(f"[get_media_metadata WARNING] {message}", file=sys.stderr) # Для локальной отладки
    def log_error(message): # Внутренняя функция для логирования ошибок
        if logger_instance: logger_instance.error(message)
        # else: print(f"[get_media_metadata ERROR] {message}", file=sys.stderr) # Для локальной отладки

    try:
        media_path_str = str(media_path) 
        command = [
            FFPROBE_PATH, "-v", "quiet", "-print_format", "json",
            "-show_format", "-show_streams", media_path_str
        ]
        # log_debug(f"Вызов ffprobe: {' '.join(command)}") # Очень детальный лог, если нужен
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True, text=True, encoding='utf-8')
        metadata = json.loads(result.stdout)
        
        stream = next((s for s in metadata.get("streams", []) if s.get("codec_type") in ["video", "audio"]), None)
        if not stream:
            log_warning(f"Не найден видео/аудио поток для {media_path_str}")
            return None

        duration_sec = float(stream.get("duration", metadata.get("format", {}).get("duration", 0)))
        original_filename = os.path.basename(media_path_str)
        _, original_extension = os.path.splitext(original_filename)

        meta = {
            "original_path": media_path_str,
            "original_filename": original_filename,
            "original_extension": original_extension.lower(),
            "duration_microseconds": int(duration_sec * 1_000_000),
            "type": stream.get("codec_type")
        }
        if meta["type"] == "video":
            meta["width"] = int(stream.get("width", 0))
            meta["height"] = int(stream.get("height", 0))
            audio_stream_in_video = next((s for s in metadata.get("streams", []) if s.get("codec_type") == "audio"), None)
            meta["has_audio_track"] = audio_stream_in_video is not None
        else: # audio
            meta["width"] = 0
            meta["height"] = 0
            meta["has_audio_track"] = True # Аудиофайл по определению имеет аудио
        return meta
    except subprocess.CalledProcessError as e_proc:
        stderr_output = e_proc.stderr.decode(errors='ignore') if isinstance(e_proc.stderr, bytes) else e_proc.stderr
        log_error(f"Ошибка вызова ffprobe для {media_path_str}: {e_proc}. Stderr: {stderr_output}")
        return None
    except json.JSONDecodeError as e_json:
        log_error(f"Ошибка декодирования JSON от ffprobe для {media_path_str}: {e_json}. Stdout: {result.stdout if 'result' in locals() else 'N/A'}")
        return None
    except Exception as e:
        log_error(f"Общая ошибка get_media_metadata для {media_path_str}: {e}")
        return None


def new_uuid() -> str:
    return str(uuid.uuid4()).upper()

def ms_to_us(ms: int) -> int:
    return int(ms * 1000)

def _generate_capcut_project_logic( 
    project_name: str,
    media_items_info: List[Dict],
    base_template_project_path: Union[str, Path],
    output_project_root_path: Union[str, Path],
    default_fps: float = 30.0,
    project_canvas_width: Optional[int] = None,
    project_canvas_height: Optional[int] = None,
    orientation: Optional[str] = None, # "vertical", "horizontal", "square" или None
    logger_instance = None # Экземпляр логгера (например, от MCP)
) -> bool:
    """
    Основная логика генерации проекта CapCut.
    Эта функция не должна напрямую выводить в stdout, если используется в MCP.
    Для логирования используется переданный logger_instance.
    """
    
    # Внутренние функции для логирования через переданный logger_instance
    def log_debug(message):
        if logger_instance: logger_instance.debug(f"[Logic] {message}")
        # else: print(f"[LOGIC_DEBUG] {message}", file=sys.stderr) # Для локальной отладки без логгера
    def log_info(message):
        if logger_instance: logger_instance.info(f"[Logic] {message}")
        # else: print(f"[LOGIC_INFO] {message}", file=sys.stderr)
    def log_warning(message):
        if logger_instance: logger_instance.warning(f"[Logic] {message}")
        # else: print(f"[LOGIC_WARNING] {message}", file=sys.stderr)
    def log_error(message):
        if logger_instance: logger_instance.error(f"[Logic] {message}")
        # else: print(f"[LOGIC_ERROR] {message}", file=sys.stderr)

    log_info(f"Начинаем генерацию проекта: '{project_name}'")
    
    base_template_project_path = Path(base_template_project_path)
    output_project_root_path = Path(output_project_root_path)

    # --- Обработка ориентации и определение размеров холста ---
    # Сначала сохраним переданные project_canvas_width/height, чтобы orientation мог их переопределить
    final_canvas_width = project_canvas_width
    final_canvas_height = project_canvas_height

    if orientation:
        orientation_lower = orientation.lower()
        if final_canvas_width is not None or final_canvas_height is not None:
            log_warning(f"Задана 'orientation' ('{orientation}'), она переопределит project_canvas_width/height.")
        
        if orientation_lower == "vertical":
            final_canvas_width = 1080
            final_canvas_height = 1920
        elif orientation_lower == "horizontal":
            final_canvas_width = 1920
            final_canvas_height = 1080
        elif orientation_lower == "square": 
            final_canvas_width = 1080
            final_canvas_height = 1080
        else:
            log_warning(f"Неизвестная ориентация '{orientation}'. Используются переданные project_canvas_width/height или значения по умолчанию/из шаблона.")
            # final_canvas_width/height остаются как были переданы или None
    
    log_debug(f"  Предварительные размеры холста: Ширина={final_canvas_width}, Высота={final_canvas_height} (до загрузки draft_info)")

    # --- Создание структуры папок проекта ---
    sanitized_project_name = project_name.replace(" ", "_").replace("/", "_").replace("\\", "_")
    new_project_full_path = output_project_root_path / sanitized_project_name
    project_resources_local_path = new_project_full_path / "Resources" / "local"
    
    try:
        os.makedirs(project_resources_local_path, exist_ok=True)
    except OSError as e:
        log_error(f"Не удалось создать директорию для ресурсов проекта '{project_resources_local_path}': {e}")
        return False
    log_info(f"  Проект будет создан в '{new_project_full_path.resolve()}'")

    # --- Определение путей к JSON файлам проекта ---
    new_draft_meta_info_path = new_project_full_path / "draft_meta_info.json"
    new_draft_info_path = new_project_full_path / "draft_info.json"
    new_draft_virtual_store_path = new_project_full_path / "draft_virtual_store.json"

    # --- Шаг 1: Копирование файлов шаблона ---
    template_files_to_copy = [
        ("draft_meta_info.json", new_draft_meta_info_path), 
        ("draft_info.json", new_draft_info_path),
        ("draft_virtual_store.json", new_draft_virtual_store_path), 
        ("draft_cover.jpg", new_project_full_path / "draft_cover.jpg") # draft_cover.jpg не JSON, но копируем
    ]
    log_debug("  Шаг 1: Копирование файлов шаблона...")
    for file_name_in_template, destination_path in template_files_to_copy:
        src = base_template_project_path / file_name_in_template
        if src.exists():
            try:
                shutil.copy2(src, destination_path)
                # log_debug(f"    Скопирован '{src}' в '{destination_path}'") # Детальное логирование
            except Exception as e:
                log_warning(f"    Не удалось скопировать '{file_name_in_template}' из '{src}' в '{destination_path}': {e}")
        else:
            if file_name_in_template != "draft_cover.jpg": # Обложка может отсутствовать
                 log_warning(f"    Файл шаблона '{src}' не найден, пропускаем копирование.")

    # --- Внутренние функции для загрузки JSON и получения структур по умолчанию ---
    def load_json_or_default(file_path: Path, default_content_factory, file_description: str = ""):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                loaded_data = json.load(f)
                # log_debug(f"    Успешно загружен {file_description} из '{file_path}'")
                return loaded_data
        except FileNotFoundError:
            log_warning(f"    Файл {file_description} '{file_path}' не найден после копирования (или не был скопирован), используется структура по умолчанию.")
            return default_content_factory()
        except json.JSONDecodeError as e_json_load:
            log_warning(f"    Файл {file_description} '{file_path}' поврежден (ошибка JSON: {e_json_load}), используется структура по умолчанию.")
            return default_content_factory()
        except Exception as e_load_generic:
            log_error(f"    Неожиданная ошибка при загрузке {file_description} из '{file_path}': {e_load_generic}. Используется структура по умолчанию.")
            return default_content_factory()

    def get_default_draft_meta_info():
        return {"cloud_package_completed_time":"","draft_cloud_capcut_purchase_info":"","draft_cloud_last_action_download":False,"draft_cloud_materials":[],"draft_cloud_package_type":"","draft_cloud_purchase_info":"","draft_cloud_template_id":"","draft_cloud_tutorial_info":"","draft_cloud_videocut_purchase_info":"","draft_cover":"draft_cover.jpg","draft_deeplink_url":"","draft_enterprise_info":{"draft_enterprise_extra":"","draft_enterprise_id":"","draft_enterprise_name":"","enterprise_material":[]},"draft_fold_path":"","draft_id":"","draft_is_ai_packaging_used":False,"draft_is_ai_shorts":False,"draft_is_ai_translate":False,"draft_is_article_video_draft":False,"draft_is_from_deeplink":"false","draft_is_invisible":False,"draft_materials":[{"type":0,"value":[]},{"type":1,"value":[]},{"type":2,"value":[]},{"type":3,"value":[]},{"type":6,"value":[]},{"type":7,"value":[]},{"type":8,"value":[]}],"draft_materials_copied_info":[],"draft_name":"","draft_need_rename_folder":False,"draft_new_version":"","draft_removable_storage_device":"","draft_root_path":"","draft_segment_extra_info":[],"draft_timeline_materials_size_":0,"draft_type":"","tm_draft_cloud_completed":"","tm_draft_cloud_modified":0,"tm_draft_create":0,"tm_draft_modified":0,"tm_draft_removed":0,"tm_duration":0}
    def get_default_draft_info(current_fps, canvas_w, canvas_h): # Принимает рассчитанные размеры
        # Используем рассчитанные размеры, если они есть, иначе дефолтные
        width_to_use = canvas_w if canvas_w is not None else 1920
        height_to_use = canvas_h if canvas_h is not None else 1080
        ratio_to_use = "custom" if canvas_w is not None and canvas_h is not None else "16:9"

        return {"canvas_config":{"background":None,"height":height_to_use,"ratio":ratio_to_use,"width":width_to_use},"color_space":0,"config":{"adjust_max_index":1,"attachment_info":[],"combination_max_index":1,"export_range":None,"extract_audio_last_index":1,"lyrics_recognition_id":"","lyrics_sync":True,"lyrics_taskinfo":[],"maintrack_adsorb":True,"material_save_mode":0,"multi_language_current":"none","multi_language_list":[],"multi_language_main":"none","multi_language_mode":"none","original_sound_last_index":1,"record_audio_last_index":1,"sticker_max_index":1,"subtitle_keywords_config":None,"subtitle_recognition_id":"","subtitle_sync":True,"subtitle_taskinfo":[],"system_font_list":[],"video_mute":False,"zoom_info_params":None},"cover":None,"create_time":0,"duration":0,"extra_info":None,"fps":float(current_fps),"free_render_index_mode_on":False,"group_container":None,"id":"","is_drop_frame_timecode":False,"keyframe_graph_list":[],"keyframes":{"adjusts":[],"audios":[],"effects":[],"filters":[],"handwrites":[],"stickers":[],"texts":[],"videos":[]},"last_modified_platform":{},"lyrics_effects":[],"materials":{"adjust_masks":[],"ai_translates":[],"audio_balances":[],"audio_effects":[],"audio_fades":[],"audio_track_indexes":[],"audios":[],"beats":[],"canvases":[],"chromas":[],"color_curves":[],"digital_humans":[],"drafts":[],"effects":[],"flowers":[],"green_screens":[],"handwrites":[],"hsl":[],"images":[],"log_color_wheels":[],"loudnesses":[],"manual_deformations":[],"masks":[],"material_animations":[],"material_colors":[],"multi_language_refs":[],"placeholders":[],"plugin_effects":[],"primary_color_wheels":[],"realtime_denoises":[],"shapes":[],"smart_crops":[],"smart_relights":[],"sound_channel_mappings":[],"speeds":[],"stickers":[],"tail_leaders":[],"text_templates":[],"texts":[],"time_marks":[],"transitions":[],"video_effects":[],"video_trackings":[],"videos":[],"vocal_beautifys":[],"vocal_separations":[]},"mutable_config":None,"name":"","new_version":"121.0.0","path":"","platform":{},"relationships":[],"render_index_track_mode_on":True,"retouch_cover":None,"source":"default","static_cover_image_path":"","time_marks":None,"tracks":[],"update_time":0,"version":360000}
    def get_default_platform_info(app_version="4.8.3", os_name="mac", os_ver="14.4"):
         return {"app_id": 359289, "app_source": "cc", "app_version": app_version,"device_id": new_uuid().replace("-","").lower(),"hard_disk_id": new_uuid().replace("-","").lower(),"mac_address": "00:00:00:00:00:00".replace(":", "").lower() + new_uuid()[:6].lower(),"os": os_name, "os_version": os_ver}
    def get_default_virtual_store():
        return {"draft_materials":[],"draft_virtual_store":[{"type":0,"value":[]},{"type":1,"value":[]},{"type":2,"value":[]}]}

    # --- Шаг 2: Загрузка JSON данных ---
    log_debug("  Шаг 2: Загрузка данных из JSON файлов...")
    meta_info_data = load_json_or_default(new_draft_meta_info_path, get_default_draft_meta_info, "draft_meta_info")
    # Важно: draft_info_data должен быть загружен ДО того, как мы попытаемся изменить его canvas_config на основе final_canvas_width/height
    draft_info_data = load_json_or_default(new_draft_info_path, 
                                           lambda: get_default_draft_info(default_fps, final_canvas_width, final_canvas_height), 
                                           "draft_info")
    virtual_store_data = load_json_or_default(new_draft_virtual_store_path, get_default_virtual_store, "draft_virtual_store")
    
    # --- Заполнение platform и last_modified_platform, если отсутствуют ---
    if not draft_info_data.get("platform") or not draft_info_data.get("platform",{}).get("app_id"):
        draft_info_data["platform"] = get_default_platform_info()
    if not draft_info_data.get("last_modified_platform") or not draft_info_data.get("last_modified_platform",{}).get("app_id"):
        lmp_platform_info = get_default_platform_info()
        lmp_platform_info["hard_disk_id"] = new_uuid().replace("-","").lower() # Убедимся, что ID диска другой
        draft_info_data["last_modified_platform"] = lmp_platform_info

    # --- Применение рассчитанных размеров холста к draft_info_data ---
    # Это нужно делать ПОСЛЕ загрузки draft_info_data, чтобы переопределить значения из шаблона/умолчания
    if final_canvas_width is not None and final_canvas_height is not None:
        log_debug(f"  Применение размеров холста к draft_info: {final_canvas_width}x{final_canvas_height}")
        if "canvas_config" not in draft_info_data or draft_info_data["canvas_config"] is None:
            draft_info_data["canvas_config"] = {} # Создаем, если полностью отсутствует
        
        draft_info_data["canvas_config"]["width"] = int(final_canvas_width)
        draft_info_data["canvas_config"]["height"] = int(final_canvas_height)
        draft_info_data["canvas_config"]["ratio"] = "custom" 
        
        if "background" not in draft_info_data["canvas_config"]: # Убедимся, что background есть
            draft_info_data["canvas_config"]["background"] = None
    else:
        # Если final_canvas_width/height не были установлены (т.е. orientation не подошел и не было явных размеров),
        # то размеры будут те, что уже есть в draft_info_data (из шаблона или из get_default_draft_info)
        loaded_w = draft_info_data.get("canvas_config", {}).get("width")
        loaded_h = draft_info_data.get("canvas_config", {}).get("height")
        log_debug(f"  Размеры холста не были явно переопределены, используются из draft_info: {loaded_w}x{loaded_h}")
    
    # --- Переменные для работы с материалами ---
    meta_materials_map = {} 
    timeline_segments_map = {}
    processed_import_time_ms_stamps = set()

    # --- Шаг 3: Обновление основных полей draft_meta_info и draft_info ---
    log_debug("  Шаг 3: Обновление основных полей проектов...")
    project_draft_id_meta = new_uuid()
    meta_info_data["draft_id"] = project_draft_id_meta
    meta_info_data["draft_name"] = project_name
    meta_info_data["draft_fold_path"] = str(new_project_full_path) 
    meta_info_data["draft_root_path"] = str(output_project_root_path) 
    current_time_us_exact = int(time.time() * 1_000_000)
    meta_info_data["tm_draft_create"] = current_time_us_exact
    meta_info_data["tm_draft_modified"] = current_time_us_exact
    
    project_draft_id_info = new_uuid() # ID проекта в draft_info (может отличаться от meta_info_data.draft_id)
    draft_info_data["id"] = project_draft_id_info 
    draft_info_data["name"] = project_name # Имя проекта также здесь
    draft_info_data["create_time"] = 0 # Оставим 0, как в примерах CapCut
    draft_info_data["update_time"] = 0 # Аналогично
    draft_info_data["fps"] = float(default_fps)
    log_debug(f"    meta_info_data.draft_id установлен в: {project_draft_id_meta}")
    log_debug(f"    draft_info_data.id установлен в: {project_draft_id_info}")


    # --- Подготовка списка материалов для draft_meta_info ---
    new_draft_materials_type_0_value = []
    # Сначала сохраняем ВСЕ не-видео/музыка материалы из шаблона (если есть)
    type_0_group_meta_template = next((g for g in meta_info_data.get("draft_materials", []) if g.get("type") == 0), None)
    if type_0_group_meta_template and "value" in type_0_group_meta_template:
        for item in type_0_group_meta_template["value"]:
            if item.get("metetype") not in ["video", "music"]:
                new_draft_materials_type_0_value.append(item)
                if "import_time_ms" in item:
                    processed_import_time_ms_stamps.add(item["import_time_ms"])
    
    current_timestamp_s_import = int(time.time())
    base_import_time_ms_for_capcut = int(time.time() * 1_000_000) # Микросекунды для import_time_ms

    # --- Шаг 4: Обработка и копирование новых медиа-элементов ---
    log_debug(f"  Шаг 4: Обработка {len(media_items_info)} медиа-элементов для проекта...")
    for item_idx, item_info_for_timeline in enumerate(media_items_info):
        # log_debug(f"    Обработка медиа #{item_idx + 1}: {item_info_for_timeline.get('path')}")
        abs_media_path = Path(item_info_for_timeline['path']).resolve() 
        media_meta_data_from_ffprobe = get_media_metadata(abs_media_path, logger_instance=logger_instance)
        if not media_meta_data_from_ffprobe:
            log_warning(f"      Пропуск медиа (нет метаданных от ffprobe): {abs_media_path}")
            continue

        new_internal_filename_stem = new_uuid().replace("-", "") # UUID без дефисов
        new_internal_filename = new_internal_filename_stem + media_meta_data_from_ffprobe["original_extension"]
        new_internal_file_path_in_project = project_resources_local_path / new_internal_filename
        
        try:
            shutil.copy2(abs_media_path, new_internal_file_path_in_project)
            # log_debug(f"      Скопирован '{abs_media_path}' в '{new_internal_file_path_in_project}'")
        except Exception as e:
            log_error(f"      Ошибка при копировании файла '{abs_media_path}' в '{new_internal_file_path_in_project}': {e}")
            continue # Пропускаем этот файл, если не удалось скопировать
        
        meta_material_id = new_uuid()
        unique_import_time_ms = base_import_time_ms_for_capcut
        while unique_import_time_ms in processed_import_time_ms_stamps: unique_import_time_ms += 1
        processed_import_time_ms_stamps.add(unique_import_time_ms)
        metetype_for_capcut = "video" if media_meta_data_from_ffprobe["type"] == "video" else "music"

        material_object_meta = {
            "create_time": current_timestamp_s_import, "duration": media_meta_data_from_ffprobe["duration_microseconds"],
            "extra_info": media_meta_data_from_ffprobe["original_filename"], "file_Path": str(new_internal_file_path_in_project), 
            "height": media_meta_data_from_ffprobe["height"], "id": meta_material_id,
            "import_time": current_timestamp_s_import, "import_time_ms": unique_import_time_ms, 
            "item_source": 1, "md5": "", "metetype": metetype_for_capcut,
            "roughcut_time_range": {"duration": media_meta_data_from_ffprobe["duration_microseconds"], "start": 0},
            "sub_time_range": {"duration": -1, "start": -1}, "type": 0, "width": media_meta_data_from_ffprobe["width"]
        }
        new_draft_materials_type_0_value.append(material_object_meta)
        meta_materials_map[meta_material_id] = {
            **media_meta_data_from_ffprobe, "internal_path": str(new_internal_file_path_in_project),
            "meta_material_id": meta_material_id, "timeline_info": item_info_for_timeline 
        }
    
    # Обновление или создание группы type:0 в meta_info_data.draft_materials
    type_0_group_meta_to_update = next((g for g in meta_info_data.get("draft_materials", []) if g.get("type") == 0), None)
    if type_0_group_meta_to_update:
        type_0_group_meta_to_update["value"] = new_draft_materials_type_0_value
    else: 
        type_0_group_meta_to_update = {"type": 0, "value": new_draft_materials_type_0_value}
        if "draft_materials" not in meta_info_data or not isinstance(meta_info_data["draft_materials"], list):
            meta_info_data["draft_materials"] = [] # Инициализируем, если отсутствует
        meta_info_data["draft_materials"].insert(0, type_0_group_meta_to_update) # Вставляем в начало
    
    # Гарантируем наличие остальных стандартных групп (type:1, 2 и т.д.)
    required_types_in_meta_materials = {1, 2, 3, 6, 7, 8} 
    existing_types_in_meta_materials = {g.get("type") for g in meta_info_data.get("draft_materials", [])}
    for req_type in required_types_in_meta_materials:
        if req_type not in existing_types_in_meta_materials:
            meta_info_data.get("draft_materials", []).append({"type": req_type, "value": []})
    meta_info_data.get("draft_materials", []).sort(key=lambda x: x.get("type", float('inf'))) # Сортировка

    # --- Шаг 5: Обновление draft_info.json (секции materials и tracks) ---
    log_debug("  Шаг 5: Обновление draft_info.json (материалы и треки)...")
    # Очищаем старые материалы и треки, чтобы избежать конфликтов ID
    for key in ["videos", "audios", "speeds", "canvases", "sound_channel_mappings", "vocal_separations", "beats"]:
        if "materials" in draft_info_data and isinstance(draft_info_data["materials"], dict) and key in draft_info_data["materials"]:
            draft_info_data["materials"][key] = []
    if "tracks" in draft_info_data: draft_info_data["tracks"] = []
    if "keyframes" in draft_info_data and isinstance(draft_info_data["keyframes"], dict):
        for key in draft_info_data["keyframes"]: draft_info_data["keyframes"][key] = []

    max_project_duration_us = 0
    tracks_data_temp = {} # { 'video_0': [segment1, ...], 'audio_0': [segment2, ...] }

    for meta_material_id, media_full_info in meta_materials_map.items():
        info_material_id = new_uuid() # ID для draft_info.materials
        timeline_info = media_full_info["timeline_info"]
        material_type_in_timeline = timeline_info["type"] # 'video' или 'audio' из входных данных

        base_material_info_entry = {
            "id": info_material_id, "local_material_id": meta_material_id,
            "path": media_full_info["internal_path"], "duration": media_full_info["duration_microseconds"],
            "category_name": "local", "check_flag": 1, 
        }

        if material_type_in_timeline == "video":
            video_material_entry = {**base_material_info_entry, "type": "video","width": media_full_info["width"], "height": media_full_info["height"],"has_audio": media_full_info.get("has_audio_track", True),"material_name": media_full_info["original_filename"],"crop_ratio":"free", "crop_scale":1.0, "extra_type_option":0, "is_unified_beauty_mode":False,"crop":{"lower_left_x":0.0,"lower_left_y":1.0,"lower_right_x":1.0,"lower_right_y":1.0,"upper_left_x":0.0,"upper_left_y":0.0,"upper_right_x":1.0,"upper_right_y":0.0},"matting":{"flag":0,"has_use_quick_brush":False,"has_use_quick_eraser":False,"interactiveTime":[],"path":"","strokes":[]},"stable":{"matrix_path":"","stable_level":0,"time_range":{"duration":0,"start":0}},"video_algorithm":{"algorithms":[],"path":""}}
            draft_info_data["materials"]["videos"].append(video_material_entry)
        elif material_type_in_timeline == "audio":
            audio_material_entry = {**base_material_info_entry, "type": "extract_music","name": media_full_info["original_filename"],"music_id": new_uuid(), "wave_points": [], "check_flag":1 }
            draft_info_data["materials"]["audios"].append(audio_material_entry)
        
        # Сопутствующие материалы (speeds, sound_channel_mappings, etc.)
        speed_id = new_uuid()
        draft_info_data["materials"]["speeds"].append({"id": speed_id, "speed": 1.0, "mode": 0, "curve_speed": None, "type": "speed"})
        sound_channel_map_id = new_uuid()
        draft_info_data["materials"]["sound_channel_mappings"].append({"id": sound_channel_map_id, "type": "none", "audio_channel_mapping": 0, "is_config_open":False})
        vocal_separation_id = new_uuid()
        draft_info_data["materials"]["vocal_separations"].append({"id": vocal_separation_id, "type": "vocal_separation", "choice":0, "production_path":"", "removed_sounds":[], "time_range":None})
        extra_material_refs = [speed_id, sound_channel_map_id, vocal_separation_id]

        if material_type_in_timeline == "video":
            canvas_id = new_uuid() # Для каждого видео-сегмента свой canvas_color (как в примере)
            draft_info_data["materials"]["canvases"].append({"id": canvas_id, "type": "canvas_color", "color": "", "blur":0.0, "image":"","image_id":"","image_name":"","source_platform":0,"team_id":""})
            extra_material_refs.insert(1, canvas_id) # Порядок: speed, canvas, sound_channel, vocal_sep
        elif material_type_in_timeline == "audio":
            beat_id = new_uuid() # Для каждого аудио-сегмента свой beats
            draft_info_data["materials"]["beats"].append({"id": beat_id, "type": "beats", "mode": 404, "gear": 404, "gear_count":0,"enable_ai_beats": False, "user_beats": [], "user_delete_ai_beats":None,"ai_beats": {"beats_path":"","beats_url":"","melody_path":"","melody_url":"","melody_percents":[0.0],"beat_speed_infos":[]}})
            extra_material_refs.insert(1, beat_id) # Порядок: speed, beats, sound_channel, vocal_sep

        # Сегмент на таймлайне
        segment_id = new_uuid()
        source_start_us = ms_to_us(timeline_info["source_start_ms"])
        timeline_duration_us = ms_to_us(timeline_info["timeline_duration_ms"])
        actual_source_duration_us = min(timeline_duration_us, media_full_info["duration_microseconds"] - source_start_us)
        actual_source_duration_us = max(0, actual_source_duration_us) # Не может быть отрицательной
        target_start_us = ms_to_us(timeline_info["timeline_start_ms"])
        
        segment_info = {"id": segment_id, "material_id": info_material_id,"source_timerange": {"start": source_start_us, "duration": actual_source_duration_us},"target_timerange": {"start": target_start_us, "duration": timeline_duration_us},"speed": 1.0, "volume": 1.0, "extra_material_refs": extra_material_refs, "render_index": 0,"clip": {"alpha": 1.0, "flip": {"horizontal": False, "vertical": False}, "rotation": 0.0, "scale": {"x": 1.0, "y": 1.0}, "transform": {"x": 0.0, "y": 0.0}},"enable_adjust": True if material_type_in_timeline == "video" else False,"enable_color_curves": True if material_type_in_timeline == "video" else False,"enable_color_wheels": True if material_type_in_timeline == "video" else False,"enable_lut": True if material_type_in_timeline == "video" else False,"intensifies_audio": False, "is_placeholder": False, "is_tone_modify": False,"last_nonzero_volume": 1.0, "reverse": False, "template_scene": "default","track_attribute":0, "visible":True}
        if material_type_in_timeline == "video":
            segment_info["uniform_scale"] = {"on":True, "value":1.0}
            segment_info["hdr_settings"] = {"intensity":1.0,"mode":1,"nits":1000}
        
        timeline_segments_map[segment_id] = {"segment_id": segment_id, "meta_material_id": meta_material_id,"original_filename": media_full_info["original_filename"]}
        track_key = f"{material_type_in_timeline}_{timeline_info['track_index']}" # e.g., "video_0", "audio_1"
        if track_key not in tracks_data_temp: tracks_data_temp[track_key] = []
        tracks_data_temp[track_key].append(segment_info)
        max_project_duration_us = max(max_project_duration_us, target_start_us + timeline_duration_us)

    # Формирование треков
    track_render_index_counter = 0
    # Сортировка ключей треков: сначала видео, потом аудио, затем по числовому индексу
    sorted_track_keys = sorted(tracks_data_temp.keys(), key=lambda k: (k.split('_')[0] != 'video', int(k.split('_')[1]))) 

    for track_key in sorted_track_keys:
        segments_for_track = tracks_data_temp[track_key]
        track_type_str, _ = track_key.split("_") # "video" or "audio"
        segments_for_track.sort(key=lambda s: s["target_timerange"]["start"]) # Сортировка сегментов внутри трека по времени начала
        for seg in segments_for_track: seg["track_render_index"] = track_render_index_counter # Присваиваем track_render_index
        
        track_entry = {"id": new_uuid(), "type": track_type_str, "segments": segments_for_track,"attribute": 0, "flag": 0, "is_default_name":True, "name":""}
        draft_info_data["tracks"].append(track_entry)
        track_render_index_counter +=1 # Увеличиваем для следующего трека

    draft_info_data["duration"] = max_project_duration_us
    meta_info_data["tm_duration"] = max_project_duration_us # Общая длительность проекта

    # --- Шаг 6: Обновление draft_virtual_store.json ---
    log_debug("  Шаг 6: Обновление draft_virtual_store.json...")
    new_virtual_store_type_0_value = []
    for meta_id, m_info in meta_materials_map.items():
        vs_entry = {"creation_time":0, "display_name":"", "filter_type":0, "id": meta_id,"import_time":0, "import_time_us":0, "sort_sub_type":0,"sort_type":0}
        new_virtual_store_type_0_value.append(vs_entry)
    
    vs_type_0_group_to_update = next((g for g in virtual_store_data.get("draft_virtual_store", []) if g.get("type") == 0), None)
    if vs_type_0_group_to_update: vs_type_0_group_to_update["value"] = new_virtual_store_type_0_value
    else:
        vs_type_0_group_to_update = {"type": 0, "value": new_virtual_store_type_0_value}
        if "draft_virtual_store" not in virtual_store_data: virtual_store_data["draft_virtual_store"] = []
        virtual_store_data["draft_virtual_store"].insert(0, vs_type_0_group_to_update)
    
    new_virtual_store_type_1_value = []
    for meta_id in meta_materials_map.keys(): new_virtual_store_type_1_value.append({"child_id": meta_id, "parent_id": ""})
    
    vs_type_1_group_to_update = next((g for g in virtual_store_data.get("draft_virtual_store", []) if g.get("type") == 1), None)
    if vs_type_1_group_to_update: vs_type_1_group_to_update["value"] = new_virtual_store_type_1_value
    else:
        vs_type_1_group_to_update = {"type": 1, "value": new_virtual_store_type_1_value}
        if "draft_virtual_store" not in virtual_store_data: virtual_store_data["draft_virtual_store"] = []
        inserted = False
        for i, g in enumerate(virtual_store_data["draft_virtual_store"]):
            if g.get("type", -1) > 1 : # Вставляем перед первой группой с type > 1
                virtual_store_data["draft_virtual_store"].insert(i, vs_type_1_group_to_update)
                inserted = True; break
        if not inserted: virtual_store_data["draft_virtual_store"].append(vs_type_1_group_to_update)
        
    # Гарантируем наличие type:2 (обычно пустой)
    if not any(g.get("type") == 2 for g in virtual_store_data.get("draft_virtual_store", [])):
        virtual_store_data.get("draft_virtual_store", []).append({"type":2, "value":[]})
    virtual_store_data.get("draft_virtual_store", []).sort(key=lambda x: x.get("type", float('inf'))) # Сортировка

    # --- Шаг 7: Сохранение всех JSON файлов ---
    log_info("  Шаг 7: Сохранение JSON файлов проекта...")
    try:
        data_to_save = [
            (new_draft_meta_info_path, meta_info_data, "draft_meta_info"),
            (new_draft_info_path, draft_info_data, "draft_info"),
            (new_draft_virtual_store_path, virtual_store_data, "draft_virtual_store")
        ]
        for path_to_save, data_object, description in data_to_save:
            # log_debug(f"    Сохранение {description} в '{path_to_save}'")
            # Для очень детальной отладки можно раскомментировать вывод содержимого data_object
            # if description == "draft_meta_info":
            #     log_debug(f"      Содержимое draft_meta_info для сохранения (draft_id): {data_object.get('draft_id')}")
            #     type_0_val = next((m.get("value") for m in data_object.get("draft_materials", []) if m.get("type")==0), [])
            #     log_debug(f"      Количество материалов в type=0 для сохранения: {len(type_0_val)}")
            #     if type_0_val: log_debug(f"      Первый материал file_Path для сохранения: {type_0_val[0].get('file_Path')}")

            with open(path_to_save, 'w', encoding='utf-8') as f:
                json.dump(data_object, f, indent=2, ensure_ascii=False)
            log_info(f"    Успешно сохранен: {path_to_save.name}") # Выводим только имя файла для краткости
    except Exception as e:
        log_error(f"  Критическая ошибка при сохранении JSON файлов: {e}")
        return False # Важно вернуть False при ошибке сохранения
            
    log_info(f"--- [_generate_capcut_project_logic] Генерация проекта '{project_name}' успешно завершена ---")
    return True