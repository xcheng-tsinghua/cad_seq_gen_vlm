from __future__ import annotations

DATASET_FINAL_SNAPSHOT = "final_snapshot.png"
DATASET_PREV_DEPTH_MAP = "prev_depth_map_with_edge.png"
DATASET_CURRENT_DEPTH_MAP = "current_depth_map_with_edge.png"
DATASET_OPERATION_PARAM = "operation_param.json"
DATASET_OVERLAYED_ALL = "modeling_preview.png"

REQUIRED_STEP_FILES = (
    DATASET_PREV_DEPTH_MAP,
    DATASET_CURRENT_DEPTH_MAP,
    DATASET_OPERATION_PARAM,
    DATASET_OVERLAYED_ALL,
)

OUTPUT_OPERATION_TYPE = "operation_type.txt"
OUTPUT_PROMPT = "prompt.txt"
OUTPUT_RETRIEVED_EXAMPLES = "retrieved_examples.json"
OUTPUT_RESPONSE = "response.json"
OUTPUT_RAW_TEXT = "raw_text.txt"
OUTPUT_GENERATED_IMAGE = "generated_image.png"
OUTPUT_GENERATED_IMAGES_DIR = "generated_images"
OUTPUT_EMU35_EVENTS_DEBUG = "emu35_events_debug.json"
OUTPUT_GENERATED_DEPTH_MAP = "generated_depth_map.png"

KB_ITEMS = "kb_items.jsonl"
KB_EMBEDDINGS = "embeddings.npy"
KB_VECTOR_STORE_META = "vector_store_meta.json"
KB_BUILD_REPORT = "build_report.json"
KB_FAISS_INDEX = "faiss.index"

MANIFEST_ALL = "manifest_all.jsonl"
MANIFEST_TRAIN = "train.jsonl"
MANIFEST_VAL = "val.jsonl"
MANIFEST_TEST = "test.jsonl"
MANIFEST_STATS = "stats.json"
MANIFEST_ISSUES = "issues.jsonl"


def generated_image_sequence_name(index: int) -> str:
    return f"image_{index:03d}.png"


def api_input_image_name(index: int) -> str:
    return f"input_{index:03d}.png"
