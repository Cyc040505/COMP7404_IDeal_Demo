"""
Data loader for IDeal demo.
Loads scene metadata, precomputed features, and interactive text data
from temp_results for both auto-replay and manual modes.
"""
import json
import numpy as np
import os
from PIL import Image


# Base directory: IDeal/demo/ -> parent is IDeal/
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_TEMP_RESULTS = os.path.join(_PROJECT_ROOT, "temp_results", "8_9", "self_scanrefer")
_DATA_SPLIT = os.path.join(_PROJECT_ROOT, "lib", "data", "split")
_DATA_TEXT = os.path.join(_PROJECT_ROOT, "lib", "data", "text", "scanrefer")
_SCENE_IMAGES_DIR = os.path.join(_PROJECT_ROOT, "demo", "scenes")


def _load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def get_scene_list():
    """Return sorted list of all 141 val scene IDs."""
    val_path = os.path.join(_DATA_SPLIT, "ScanRefer_filtered_val.txt")
    with open(val_path, "r", encoding="utf-8") as f:
        scenes = [line.strip() for line in f if line.strip()]
    return sorted(scenes)


def load_interactive_data():
    """
    Load bert_interactive_text.jsonl and return dict:
        scene_id -> list of records (one per caption, 10 per scene)
    Each record has: scene_id, description, answers, tag, etps
    """
    jsonl_path = os.path.join(_TEMP_RESULTS, "bert_interactive_text.jsonl")
    records = _load_jsonl(jsonl_path)
    by_scene = {}
    for rec in records:
        sid = rec["scene_id"]
        by_scene.setdefault(sid, []).append(rec)
    return by_scene


def load_precomputed_features():
    """
    Load precomputed numpy arrays from temp_results.
    Returns:
        interactive_features: (1410, 4, 1024) — [0]=orig desc, [1:4]=answers
        tags: (1410, 6) — prompt template tags
        etps: (1410, 6) — entropy values
        summarized_features: (1410, 3, 1024) — summary embeddings
    """
    interactive_features = np.load(os.path.join(_TEMP_RESULTS, "bert_interactive_features_5.npy"))
    tags = np.load(os.path.join(_TEMP_RESULTS, "bert_tags_5.npy"))
    etps = np.load(os.path.join(_TEMP_RESULTS, "bert_etp_5.npy"))
    summarized_features = np.load(os.path.join(_TEMP_RESULTS, "summarized_features.npy"))
    return interactive_features, tags, etps, summarized_features


def get_scene_records(scene_id):
    """Get all 10 caption records for a given scene_id."""
    data = load_interactive_data()
    return data.get(scene_id, [])


def get_record_info(record_idx, scene_id=None, all_records=None):
    """
    Get information for a specific record (0-9 within a scene).
    If scene_id not provided, uses all_records.
    Returns dict with description, answers, tag, etps.
    """
    if all_records is None:
        if scene_id:
            all_records = get_scene_records(scene_id)
        else:
            all_records = load_interactive_data()
            # flatten
            all_records = [r for recs in all_records.values() for r in recs]

    if 0 <= record_idx < len(all_records):
        return all_records[record_idx]
    return None


def get_scene_images(scene_id, max_images=15):
    """
    Load scene images from demo/scenes/{scene_id}/images/.
    Returns list of PIL Image objects (up to max_images).
    """
    scene_dir = os.path.join(_SCENE_IMAGES_DIR, scene_id, "images")
    if not os.path.exists(scene_dir):
        return []

    # Get all JPG files sorted
    jpg_files = sorted([f for f in os.listdir(scene_dir) if f.endswith('.jpg')])
    if not jpg_files:
        return []

    # Load and return up to max_images
    images = []
    for jpg in jpg_files[:max_images]:
        try:
            img_path = os.path.join(scene_dir, jpg)
            img = Image.open(img_path)
            images.append(img)
        except Exception as e:
            print(f"Warning: Failed to load {img_path}: {e}")
            continue

    return images


def get_scene_image_count(scene_id):
    """Get number of available images for a scene."""
    scene_dir = os.path.join(_SCENE_IMAGES_DIR, scene_id, "images")
    if not os.path.exists(scene_dir):
        return 0
    jpg_files = [f for f in os.listdir(scene_dir) if f.endswith('.jpg')]
    return len(jpg_files)
