import json
import os
import uuid
import subprocess
import time
import shutil
from pathlib import Path 
from typing import List, Dict, Optional, Union
import sys 
import logging 
# import copy # Раскомментируй, если будешь использовать deepcopy в Шаге 7

# --- Конфигурация ---
FFPROBE_PATH = "ffprobe" 

def get_media_metadata(media_path: Union[str, Path], logger_instance=None) -> Optional[Dict]:
    def log_warning(message):
        if logger_instance: logger_instance.warning(f"[FFProbe] {message}")
    def log_error(message):
        if logger_instance: logger_instance.error(f"[FFProbe] {message}")

    try:
        media_path_str = str(media_path) 
        command = [ FFPROBE_PATH, "-v", "quiet", "-print_format", "json", "-show_format", "-show_streams", media_path_str]
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True, text=True, encoding='utf-8')
        metadata = json.loads(result.stdout)
        stream = next((s for s in metadata.get("streams", []) if s.get("codec_type") in ["video", "audio"]), None)
        if not stream: log_warning(f"Не найден видео/аудио поток для {media_path_str}"); return None
        duration_sec = float(stream.get("duration", metadata.get("format", {}).get("duration", 0)))
        original_filename = os.path.basename(media_path_str) # Получаем полное имя файла
        original_extension = os.path.splitext(original_filename)[1] # Получаем расширение, включая точку
        meta = {
            "original_path": media_path_str, "original_filename": original_filename,
            "original_extension": original_extension.lower(), "duration_microseconds": int(duration_sec * 1_000_000),
            "type": stream.get("codec_type")
        }
        if meta["type"] == "video":
            meta["width"], meta["height"] = int(stream.get("width", 0)), int(stream.get("height", 0))
            meta["has_audio_track"] = next((s for s in metadata.get("streams", []) if s.get("codec_type") == "audio"), None) is not None
        else: meta["width"], meta["height"], meta["has_audio_track"] = 0, 0, True
        return meta
    except subprocess.CalledProcessError as e: stderr_output = e.stderr.decode(errors='ignore') if isinstance(e.stderr, bytes) else e.stderr; log_error(f"Ошибка ffprobe для {media_path_str}: {e}. Stderr: {stderr_output}"); return None
    except json.JSONDecodeError as e: stdout_output = result.stdout if 'result' in locals() and hasattr(result, 'stdout') else 'N/A'; log_error(f"Ошибка JSON от ffprobe для {media_path_str}: {e}. Stdout: {stdout_output}"); return None
    except Exception as e: log_error(f"Общая ошибка get_media_metadata для {media_path_str}: {e}"); return None

def new_uuid() -> str: return str(uuid.uuid4()).upper()
def ms_to_us(ms: int) -> int: return int(ms * 1000)

def _generate_capcut_project_logic( 
    project_name: str, media_items_info: List[Dict], base_template_project_path: Union[str, Path],
    output_project_root_path: Union[str, Path], default_fps: float = 30.0,
    project_canvas_width: Optional[int] = None, project_canvas_height: Optional[int] = None,
    orientation: Optional[str] = None, logger_instance = None 
) -> bool:
    
    def log_debug(message):
        if logger_instance: logger_instance.debug(f"[Logic_DEBUG] {message}")
    def log_info(message):
        if logger_instance: logger_instance.info(f"[Logic_INFO] {message}")
    def log_warning(message):
        if logger_instance: logger_instance.warning(f"[Logic_WARNING] {message}")
    def log_error(message):
        if logger_instance: logger_instance.error(f"[Logic_ERROR] {message}")

    log_info(f"Начинаем генерацию проекта: '{project_name}'")
    base_template_project_path, output_project_root_path = Path(base_template_project_path), Path(output_project_root_path)
    final_canvas_width, final_canvas_height = project_canvas_width, project_canvas_height
    if orientation:
        orientation_lower = orientation.lower()
        if final_canvas_width is not None or final_canvas_height is not None: log_warning(f"Задана 'orientation' ('{orientation}'), она переопределит project_canvas_width/height.")
        if orientation_lower == "vertical": final_canvas_width, final_canvas_height = 1080, 1920
        elif orientation_lower == "horizontal": final_canvas_width, final_canvas_height = 1920, 1080
        elif orientation_lower == "square": final_canvas_width, final_canvas_height = 1080, 1080
        else: log_warning(f"Неизвестная ориентация '{orientation}'.")
    log_debug(f"  Предварительные размеры холста: Ширина={final_canvas_width}, Высота={final_canvas_height}")

    sanitized_project_name = project_name.replace(" ", "_").replace("/", "_").replace("\\", "_")
    new_project_full_path = output_project_root_path / sanitized_project_name
    project_resources_local_path = new_project_full_path / "Resources" / "local"
    try: os.makedirs(project_resources_local_path, exist_ok=True)
    except OSError as e: log_error(f"Не удалось создать '{project_resources_local_path}': {e}"); return False
    log_info(f"  Проект будет создан в '{new_project_full_path.resolve()}'")

    new_draft_meta_info_path, new_draft_info_path, new_draft_virtual_store_path = new_project_full_path/"draft_meta_info.json", new_project_full_path/"draft_info.json", new_project_full_path/"draft_virtual_store.json"
    template_files_to_copy = [("draft_meta_info.json",new_draft_meta_info_path),("draft_info.json",new_draft_info_path),("draft_virtual_store.json",new_draft_virtual_store_path),("draft_cover.jpg",new_project_full_path/"draft_cover.jpg")]
    log_debug("  Шаг 1: Копирование файлов шаблона...")
    for fname_template, dest_path in template_files_to_copy:
        src = base_template_project_path / fname_template
        if src.exists():
            try: shutil.copy2(src, dest_path)
            except Exception as e: log_warning(f"    Не удалось скопировать '{fname_template}': {e}")
        elif fname_template != "draft_cover.jpg": log_warning(f"    Файл шаблона '{src}' не найден.")
    
    def get_default_draft_meta_info(): return {"cloud_package_completed_time":"","draft_cloud_capcut_purchase_info":"","draft_cloud_last_action_download":False,"draft_cloud_materials":[],"draft_cloud_package_type":"","draft_cloud_purchase_info":"","draft_cloud_template_id":"","draft_cloud_tutorial_info":"","draft_cloud_videocut_purchase_info":"","draft_cover":"draft_cover.jpg","draft_deeplink_url":"","draft_enterprise_info":{"draft_enterprise_extra":"","draft_enterprise_id":"","draft_enterprise_name":"","enterprise_material":[]},"draft_fold_path":"","draft_id":"","draft_is_ai_packaging_used":False,"draft_is_ai_shorts":False,"draft_is_ai_translate":False,"draft_is_article_video_draft":False,"draft_is_from_deeplink":"false","draft_is_invisible":False,"draft_materials":[{"type":0,"value":[]},{"type":1,"value":[]},{"type":2,"value":[]},{"type":3,"value":[]},{"type":6,"value":[]},{"type":7,"value":[]},{"type":8,"value":[]}],"draft_materials_copied_info":[],"draft_name":"","draft_need_rename_folder":False,"draft_new_version":"","draft_removable_storage_device":"","draft_root_path":"","draft_segment_extra_info":[],"draft_timeline_materials_size_":0,"draft_type":"","tm_draft_cloud_completed":"","tm_draft_cloud_modified":0,"tm_draft_create":0,"tm_draft_modified":0,"tm_draft_removed":0,"tm_duration":0}
    def get_default_draft_info(fps, cw, ch): w, h, r = (cw if cw else 1920), (ch if ch else 1080), ("custom" if cw and ch else "16:9"); return {"canvas_config":{"background":None,"height":h,"ratio":r,"width":w},"color_space":0,"config":{"adjust_max_index":1,"attachment_info":[],"combination_max_index":1,"export_range":None,"extract_audio_last_index":1,"lyrics_recognition_id":"","lyrics_sync":True,"lyrics_taskinfo":[],"maintrack_adsorb":True,"material_save_mode":0,"multi_language_current":"none","multi_language_list":[],"multi_language_main":"none","multi_language_mode":"none","original_sound_last_index":1,"record_audio_last_index":1,"sticker_max_index":1,"subtitle_keywords_config":None,"subtitle_recognition_id":"","subtitle_sync":True,"subtitle_taskinfo":[],"system_font_list":[],"video_mute":False,"zoom_info_params":None},"cover":None,"create_time":0,"duration":0,"extra_info":None,"fps":float(fps),"free_render_index_mode_on":False,"group_container":None,"id":"","is_drop_frame_timecode":False,"keyframe_graph_list":[],"keyframes":{"adjusts":[],"audios":[],"effects":[],"filters":[],"handwrites":[],"stickers":[],"texts":[],"videos":[]},"last_modified_platform":{},"lyrics_effects":[],"materials":{"adjust_masks":[],"ai_translates":[],"audio_balances":[],"audio_effects":[],"audio_fades":[],"audio_track_indexes":[],"audios":[],"beats":[],"canvases":[],"chromas":[],"color_curves":[],"digital_humans":[],"drafts":[],"effects":[],"flowers":[],"green_screens":[],"handwrites":[],"hsl":[],"images":[],"log_color_wheels":[],"loudnesses":[],"manual_deformations":[],"masks":[],"material_animations":[],"material_colors":[],"multi_language_refs":[],"placeholders":[],"plugin_effects":[],"primary_color_wheels":[],"realtime_denoises":[],"shapes":[],"smart_crops":[],"smart_relights":[],"sound_channel_mappings":[],"speeds":[],"stickers":[],"tail_leaders":[],"text_templates":[],"texts":[],"time_marks":[],"transitions":[],"video_effects":[],"video_trackings":[],"videos":[],"vocal_beautifys":[],"vocal_separations":[]},"mutable_config":None,"name":"","new_version":"121.0.0","path":"","platform":{},"relationships":[],"render_index_track_mode_on":True,"retouch_cover":None,"source":"default","static_cover_image_path":"","time_marks":None,"tracks":[],"update_time":0,"version":360000}
    def get_default_platform_info(av="4.8.3", osn="mac", osv="14.4"): return {"app_id":359289,"app_source":"cc","app_version":av,"device_id":new_uuid().replace("-","").lower(),"hard_disk_id":new_uuid().replace("-","").lower(),"mac_address":"000000000000"+new_uuid()[:6].lower(),"os":osn,"os_version":osv}
    def get_default_virtual_store(): return {"draft_materials":[],"draft_virtual_store":[{"type":0,"value":[]},{"type":1,"value":[]},{"type":2,"value":[]}]}
    def load_json_or_default(fp:Path, df, fd:str=""):
        try:
            with open(fp,'r',encoding='utf-8') as f: return json.load(f)
        except FileNotFoundError: log_warning(f"    Файл {fd} '{fp}' не найден, default."); return df()
        except json.JSONDecodeError as e: log_warning(f"    Файл {fd} '{fp}' поврежден (JSON:{e}), default."); return df()
        except Exception as e: log_error(f"    Ошибка загрузки {fd} из '{fp}': {e}. Default."); return df()

    log_debug("  Шаг 2: Загрузка данных из JSON файлов...")
    meta_info_data = load_json_or_default(new_draft_meta_info_path, get_default_draft_meta_info, "meta_info")
    draft_info_data = load_json_or_default(new_draft_info_path, lambda: get_default_draft_info(default_fps, final_canvas_width, final_canvas_height), "draft_info")
    virtual_store_data = load_json_or_default(new_draft_virtual_store_path, get_default_virtual_store, "virtual_store")
    
    loaded_t0_mats_val = []
    t0_grp = next((g for g in meta_info_data.get("draft_materials",[]) if g.get("type")==0),None)
    if t0_grp and isinstance(t0_grp.get("value"),list): loaded_t0_mats_val=t0_grp["value"]
    log_debug(f"    meta_info_data: draft_materials[type=0].value ПОСЛЕ ЗАГРУЗКИ: {len(loaded_t0_mats_val)} эл.")
    if loaded_t0_mats_val: log_debug(f"      Первый загруженный: ID={loaded_t0_mats_val[0].get('id')}, Path='{loaded_t0_mats_val[0].get('file_Path')}'")

    if not draft_info_data.get("platform") or not draft_info_data.get("platform",{}).get("app_id"): draft_info_data["platform"]=get_default_platform_info()
    if not draft_info_data.get("last_modified_platform") or not draft_info_data.get("last_modified_platform",{}).get("app_id"):
        lmp=get_default_platform_info(); lmp["hard_disk_id"]=new_uuid().replace("-","").lower(); draft_info_data["last_modified_platform"]=lmp
    if final_canvas_width and final_canvas_height:
        log_debug(f"  Применение размеров холста: {final_canvas_width}x{final_canvas_height}")
        cfg=draft_info_data.setdefault("canvas_config",{}) # Используем setdefault для безопасности
        cfg["width"],cfg["height"],cfg["ratio"]=int(final_canvas_width),int(final_canvas_height),"custom"
        cfg.setdefault("background",None)
    else: log_debug(f"  Размеры холста не переопределены, используются из draft_info: {draft_info_data.get('canvas_config',{}).get('width')}x{draft_info_data.get('canvas_config',{}).get('height')}")
    
    meta_mats_map, tl_segs_map, import_time_ms_stamps = {},{},set()
    log_debug("  Шаг 3: Обновление основных полей...")
    meta_id_meta = new_uuid(); meta_info_data.update({"draft_id":meta_id_meta,"draft_name":project_name,"draft_fold_path":str(new_project_full_path),"draft_root_path":str(output_project_root_path)})
    cur_time_us = int(time.time()*1_000_000); meta_info_data.update({"tm_draft_create":cur_time_us,"tm_draft_modified":cur_time_us})
    info_id_info = new_uuid(); draft_info_data.update({"id":info_id_info,"name":project_name,"create_time":0,"update_time":0,"fps":float(default_fps)})
    log_debug(f"    meta_info_data.draft_id: {meta_id_meta}, draft_info_data.id: {info_id_info}")

    new_dm_t0_val = []
    t0_grp_load = next((g for g in meta_info_data.get("draft_materials",[]) if g.get("type")==0),None)
    if t0_grp_load and isinstance(t0_grp_load.get("value"),list):
        log_debug(f"    Фильтрация type_0 ({len(t0_grp_load['value'])} эл)...")
        for item in t0_grp_load["value"]:
            if item.get("metetype") not in ["video","music"]: new_dm_t0_val.append(item); import_time_ms_stamps.add(item.get("import_time_ms"))
    else: log_debug(f"    type_0_group_meta_from_load не найден или его 'value' не список. Фильтрация не будет произведена.")
    log_debug(f"    new_dm_t0_val ПОСЛЕ ФИЛЬТРАЦИИ шаблонных: {len(new_dm_t0_val)} элементов.")
    
    cur_ts_s_imp = int(time.time()); base_imp_t_ms_cc = int(time.time()*1_000_000)
    log_debug(f"  Шаг 4: Обработка {len(media_items_info)} медиа...")
    for item_info in media_items_info:
        abs_mp = Path(item_info['path']).resolve(); media_m = get_media_metadata(abs_mp,logger_instance)
        if not media_m: log_warning(f"      Пропуск (нет метаданных): {abs_mp}"); continue
        new_fn_stem = new_uuid().replace("-", "") # Имя файла без расширения
        new_fn = new_fn_stem + media_m["original_extension"] # Полное новое имя файла
        new_fp_proj = project_resources_local_path/new_fn
        try: shutil.copy2(abs_mp,new_fp_proj)
        except Exception as e: log_error(f"      Ошибка копирования '{abs_mp}': {e}"); continue
        meta_matid, uniq_imp_ms = new_uuid(), base_imp_t_ms_cc
        while uniq_imp_ms in import_time_ms_stamps: uniq_imp_ms+=1
        import_time_ms_stamps.add(uniq_imp_ms)
        mtype = "video" if media_m["type"]=="video" else "music"
        mat_obj = {"create_time":cur_ts_s_imp,"duration":media_m["duration_microseconds"],"extra_info":media_m["original_filename"],"file_Path":str(new_fp_proj),"height":media_m["height"],"id":meta_matid,"import_time":cur_ts_s_imp,"import_time_ms":uniq_imp_ms,"item_source":1,"md5":"","metetype":mtype,"roughcut_time_range":{"duration":media_m["duration_microseconds"],"start":0},"sub_time_range":{"duration":-1,"start":-1},"type":0,"width":media_m["width"]}
        new_dm_t0_val.append(mat_obj)
        meta_mats_map[meta_matid] = {**media_m,"internal_path":str(new_fp_proj),"meta_material_id":meta_matid,"timeline_info":item_info}
    log_debug(f"    new_dm_t0_val ПОСЛЕ ДОБАВЛЕНИЯ НОВЫХ: {len(new_dm_t0_val)} элементов.")
    if new_dm_t0_val and media_items_info : log_debug(f"      Последний добавленный: ID={new_dm_t0_val[-1].get('id')}, Path='{new_dm_t0_val[-1].get('file_Path')}'")
    
    # --- ИЗМЕНЕНИЕ ЗДЕСЬ: Формирование финального списка draft_materials ---
    final_draft_mats_list = []
    final_draft_mats_list.append({"type": 0, "value": new_dm_t0_val}) # Наша новая/обновленная группа type=0
    log_debug(f"    В final_draft_mats_list добавлена группа type=0 с {len(new_dm_t0_val)} элементами.")

    if meta_info_data.get("draft_materials"): # Только если draft_materials вообще был в шаблоне
        for group_from_template in meta_info_data.get("draft_materials", []):
            if group_from_template.get("type") != 0: # Копируем все группы, КРОМЕ type=0
                final_draft_mats_list.append(group_from_template) 
                log_debug(f"    В final_draft_mats_list скопирована группа type={group_from_template.get('type')} из meta_info_data (шаблона).")
    
    req_types_meta={1,2,3,6,7,8}; cur_types_final={g.get("type") for g in final_draft_mats_list}
    for rt in req_types_meta:
        if rt not in cur_types_final: 
            final_draft_mats_list.append({"type":rt,"value":[]}); 
            log_debug(f"    В final_draft_mats_list добавлена недостающая пустая группа type={rt}.")
            
    final_draft_mats_list.sort(key=lambda x:x.get("type",float('inf')))
    meta_info_data["draft_materials"]=final_draft_mats_list # Полностью перезаписываем draft_materials
    log_debug(f"    Финальный meta_info_data['draft_materials'] содержит {len(meta_info_data['draft_materials'])} групп.")

    final_t0_mats_val=[]
    f_t0_grp_meta=next((g for g in meta_info_data.get("draft_materials",[]) if g.get("type")==0),None)
    if f_t0_grp_meta and isinstance(f_t0_grp_meta.get("value"),list): final_t0_mats_val=f_t0_grp_meta["value"]
    log_debug(f"    meta_info_data: draft_materials[type=0].value ПЕРЕД СОХРАНЕНИЕМ: {len(final_t0_mats_val)} эл.")
    if final_t0_mats_val: 
        log_debug(f"      Первый для сохранения: ID={final_t0_mats_val[0].get('id')}, Path='{final_t0_mats_val[0].get('file_Path')}'")
        for i_item_log, item_log in enumerate(final_t0_mats_val[:3]):
             log_debug(f"        Элемент {i_item_log} для сохр: ID={item_log.get('id')}, Path='{item_log.get('file_Path')}', Extra='{item_log.get('extra_info')}'")


    log_debug("  Шаг 5: Обновление draft_info.json (материалы и треки)...")
    for k in ["videos","audios","speeds","canvases","sound_channel_mappings","vocal_separations","beats"]: draft_info_data.setdefault("materials",{}).update({k:[]})
    draft_info_data["tracks"]=[]; draft_info_data.setdefault("keyframes",{}).update({k:[] for k in draft_info_data.get("keyframes",{}).keys()})
    max_p_dur_us, trks_data = 0,{}
    for meta_mid, media_fi in meta_mats_map.items():
        inf_mid, ti, mttl = new_uuid(),media_fi["timeline_info"],media_fi["timeline_info"]["type"]
        base_me = {"id":inf_mid,"local_material_id":meta_mid,"path":media_fi["internal_path"],"duration":media_fi["duration_microseconds"],"category_name":"local","check_flag":1}
        if mttl=="video": draft_info_data["materials"]["videos"].append({**base_me,"type":"video","width":media_fi["width"],"height":media_fi["height"],"has_audio":media_fi.get("has_audio_track",True),"material_name":media_fi["original_filename"],"crop_ratio":"free","crop_scale":1.0,"extra_type_option":0,"is_unified_beauty_mode":False,"crop":{"lower_left_x":0.0,"lower_left_y":1.0,"lower_right_x":1.0,"lower_right_y":1.0,"upper_left_x":0.0,"upper_left_y":0.0,"upper_right_x":1.0,"upper_right_y":0.0},"matting":{"flag":0,"has_use_quick_brush":False,"has_use_quick_eraser":False,"interactiveTime":[],"path":"","strokes":[]},"stable":{"matrix_path":"","stable_level":0,"time_range":{"duration":0,"start":0}},"video_algorithm":{"algorithms":[],"path":""}})
        elif mttl=="audio": draft_info_data["materials"]["audios"].append({**base_me,"type":"extract_music","name":media_fi["original_filename"],"music_id":new_uuid(),"wave_points":[],"check_flag":1})
        sp_id,sc_id,vc_id=new_uuid(),new_uuid(),new_uuid()
        draft_info_data["materials"]["speeds"].append({"id":sp_id,"speed":1.0,"mode":0,"curve_speed":None,"type":"speed"})
        draft_info_data["materials"]["sound_channel_mappings"].append({"id":sc_id,"type":"none","audio_channel_mapping":0,"is_config_open":False})
        draft_info_data["materials"]["vocal_separations"].append({"id":vc_id,"type":"vocal_separation","choice":0,"production_path":"","removed_sounds":[],"time_range":None})
        xtra_refs=[sp_id,sc_id,vc_id]
        if mttl=="video": cnvs_id=new_uuid(); draft_info_data["materials"]["canvases"].append({"id":cnvs_id,"type":"canvas_color","color":"","blur":0.0,"image":"","image_id":"","image_name":"","source_platform":0,"team_id":""}); xtra_refs.insert(1,cnvs_id)
        elif mttl=="audio": bt_id=new_uuid(); draft_info_data["materials"]["beats"].append({"id":bt_id,"type":"beats","mode":404,"gear":404,"gear_count":0,"enable_ai_beats":False,"user_beats":[],"user_delete_ai_beats":None,"ai_beats":{"beats_path":"","beats_url":"","melody_path":"","melody_url":"","melody_percents":[0.0],"beat_speed_infos":[]}}); xtra_refs.insert(1,bt_id)
        sg_id,s_start_us,tl_d_us=new_uuid(),ms_to_us(ti["source_start_ms"]),ms_to_us(ti["timeline_duration_ms"])
        act_s_d_us=min(tl_d_us,media_fi["duration_microseconds"]-s_start_us); act_s_d_us=max(0,act_s_d_us)
        tgt_s_us=ms_to_us(ti["timeline_start_ms"])
        sg_info={"id":sg_id,"material_id":inf_mid,"source_timerange":{"start":s_start_us,"duration":act_s_d_us},"target_timerange":{"start":tgt_s_us,"duration":tl_d_us},"speed":1.0,"volume":1.0,"extra_material_refs":xtra_refs,"render_index":0,"clip":{"alpha":1.0,"flip":{"horizontal":False,"vertical":False},"rotation":0.0,"scale":{"x":1.0,"y":1.0},"transform":{"x":0.0,"y":0.0}},"enable_adjust":mttl=="video","enable_color_curves":mttl=="video","enable_color_wheels":mttl=="video","enable_lut":mttl=="video","intensifies_audio":False,"is_placeholder":False,"is_tone_modify":False,"last_nonzero_volume":1.0,"reverse":False,"template_scene":"default","track_attribute":0,"visible":True}
        if mttl=="video": sg_info["uniform_scale"]={"on":True,"value":1.0};sg_info["hdr_settings"]={"intensity":1.0,"mode":1,"nits":1000}
        tl_segs_map[sg_id]={"segment_id":sg_id,"meta_material_id":meta_mid,"original_filename":media_fi["original_filename"]}
        trk_k=f"{mttl}_{ti['track_index']}"; trks_data.setdefault(trk_k,[]).append(sg_info)
        max_p_dur_us=max(max_p_dur_us,tgt_s_us+tl_d_us)
    trk_r_idx_cnt=0;srt_trks=sorted(trks_data.keys(),key=lambda k:(k.split('_')[0]!='video',int(k.split('_')[1])))
    for trk_k_it in srt_trks:
        sgs_trk = trks_data[trk_k_it]
        trk_type_s = trk_k_it.split("_")[0] # Исправлено
        sgs_trk.sort(key=lambda s:s["target_timerange"]["start"])
        for s_item in sgs_trk: s_item["track_render_index"]=trk_r_idx_cnt
        trk_e={"id":new_uuid(),"type":trk_type_s,"segments":sgs_trk,"attribute":0,"flag":0,"is_default_name":True,"name":""}
        draft_info_data["tracks"].append(trk_e); trk_r_idx_cnt+=1
    draft_info_data["duration"]=max_p_dur_us; meta_info_data["tm_duration"]=max_p_dur_us

    log_debug("  Шаг 6: Обновление draft_virtual_store.json...")
    n_vs_t0_v=[{"creation_time":0,"display_name":"","filter_type":0,"id":mid,"import_time":0,"import_time_us":0,"sort_sub_type":0,"sort_type":0} for mid in meta_mats_map.keys()]
    vs_t0_g=next((g for g in virtual_store_data.get("draft_virtual_store",[]) if g.get("type")==0),None)
    if vs_t0_g: vs_t0_g["value"]=n_vs_t0_v
    else: vs_t0_g={"type":0,"value":n_vs_t0_v}; virtual_store_data.setdefault("draft_virtual_store",[]).insert(0,vs_t0_g)
    n_vs_t1_v=[{"child_id":mid,"parent_id":""} for mid in meta_mats_map.keys()]
    vs_t1_g=next((g for g in virtual_store_data.get("draft_virtual_store",[]) if g.get("type")==1),None)
    if vs_t1_g: vs_t1_g["value"]=n_vs_t1_v
    else:
        vs_t1_g={"type":1,"value":n_vs_t1_v}; virtual_store_data.setdefault("draft_virtual_store",[])
        ins_=False;
        for i,g_v in enumerate(virtual_store_data["draft_virtual_store"]):
            if g_v.get("type",-1)>1: virtual_store_data["draft_virtual_store"].insert(i,vs_t1_g); ins_=True; break
        if not ins_: virtual_store_data["draft_virtual_store"].append(vs_t1_g)
    if not any(g.get("type")==2 for g in virtual_store_data.get("draft_virtual_store",[])): virtual_store_data.get("draft_virtual_store",[]).append({"type":2,"value":[]})
    virtual_store_data.get("draft_virtual_store",[]).sort(key=lambda x:x.get("type",float('inf')))

    log_info("  Шаг 7: Сохранение JSON файлов проекта...")
    try:
        d_to_save=[(new_draft_meta_info_path,meta_info_data,"meta_info"),(new_draft_info_path,draft_info_data,"draft_info"),(new_draft_virtual_store_path,virtual_store_data,"virtual_store")]
        for pth_s, d_obj, dsc in d_to_save:
            if dsc == "meta_info": # Дополнительные логи для draft_meta_info
                log_debug(f"    [ПРОВЕРКА ПЕРЕД ЗАПИСЬЮ DRAFT_META_INFO] Объект data_object_to_save:")
                log_debug(f"      draft_id: {d_obj.get('draft_id')}")
                dmats_to_save = d_obj.get("draft_materials", [])
                t0val_to_save = []
                temp_t0_grp_s = next((g for g in dmats_to_save if g.get("type") == 0), None)
                if temp_t0_grp_s and isinstance(temp_t0_grp_s.get("value"),list): t0val_to_save = temp_t0_grp_s.get("value")
                log_debug(f"      Количество материалов в data_object_to_save.draft_materials[type=0].value: {len(t0val_to_save)}")
                if t0val_to_save: log_debug(f"        Первый для записи (ID): {t0val_to_save[0].get('id')}, Path: '{t0val_to_save[0].get('file_Path')}'")
            
            with open(pth_s,'w',encoding='utf-8') as f: json.dump(d_obj,f,indent=2,ensure_ascii=False)
            log_info(f"    Успешно сохранен: {pth_s.name}")

            if dsc == "meta_info": # Проверка после записи
                log_debug(f"    [ПРОВЕРКА ПОСЛЕ ЗАПИСИ DRAFT_META_INFO] Чтение {pth_s}...")
                try:
                    with open(pth_s, 'r', encoding='utf-8') as f_chk: content_as = json.load(f_chk)
                    ca_dmats = content_as.get("draft_materials", []); ca_t0val = []
                    ca_t0grp = next((g for g in ca_dmats if g.get("type")==0),None)
                    if ca_t0grp and isinstance(ca_t0grp.get("value"),list): ca_t0val = ca_t0grp.get("value")
                    log_debug(f"      Файл ПОСЛЕ ЗАПИСИ: draft_id={content_as.get('draft_id')}, материалов[type=0]={len(ca_t0val)}")
                    if ca_t0val: log_debug(f"        Первый из файла ПОСЛЕ ЗАПИСИ (ID): {ca_t0val[0].get('id')}, Path: '{ca_t0val[0].get('file_Path')}'")
                except Exception as e_chk: log_error(f"      Не удалось прочитать {pth_s} после записи: {e_chk}")
                
    except Exception as e: log_error(f"  Критическая ошибка при сохранении JSON: {e}"); return False
            
    log_info(f"--- [_generate_capcut_project_logic] Генерация проекта '{project_name}' успешно завершена ---")
    return True