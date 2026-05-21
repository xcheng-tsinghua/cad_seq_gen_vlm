"""Phase 2 — Autoregressive reverse modeling planner inference & simulation (MVP)."""

from __future__ import annotations

import argparse
import os
import re
import json
import logging
from typing import List, Optional, Dict

import torch
from PIL import Image

# Fixed instruction template for prompt construction
FIXED_USER_PROMPT = (
    "NPC (normalized pixel coordinate): Divide the image into 1000 equal units on both sides, and "
    "represent the position of a point in the image with an integer pair (int_x, int_y) in the range [0, 100].\n"
    "CCL-Shape (current modeling operation constructed local shape).\n"
    "Based on the part final image and the current part depth map, plan the next operation and output "
    "the following instruction in JSON format:\n"
    "{\n"
    '    "operation_type": "type",\n'
    '    "sketch (red)": [[x1,y1], ...],\n'
    '    "other_counter (green / magenta)": [[x1,y1], ...],\n'
    '    "terminate_face_contour (blue)": [[x1,y1], ...],\n'
    '    "sketch_plane_contour (yellow)": [[x1,y1], ...],\n'
    '    "reference_geom_contour (cyan)": [[x1,y1], ...],\n'
    '    "target_region_bbox (red)": [ymin, xmin, ymax, xmax]\n'
    "}"
)

logger = logging.getLogger(__name__)


def extract_json(text: str) -> dict:
    """Robust JSON extraction from LLM generation text."""
    try:
        return json.loads(text.strip())
    except Exception:
        pass
    
    # Try finding JSON wrapped in markdown code blocks
    match = re.search(r"```json\s*(\{.*?\})\s*```", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except Exception:
            pass
            
    # Try finding first balanced curly braces block
    match_braces = re.search(r"(\{.*?\})", text, re.DOTALL)
    if match_braces:
        try:
            return json.loads(match_braces.group(1))
        except Exception:
            pass
            
    return {}


class QwenMLLMPlanner:
    """Fine-tuned Qwen2.5-VL Planner inference wrapper."""

    def __init__(self, checkpoint_path: str, device: str = "cuda") -> None:
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
        
        self.device = torch.device(device)
        self.processor = AutoProcessor.from_pretrained(checkpoint_path)
        
        torch_dtype = torch.bfloat16 if self.device.type == "cuda" else torch.float32
        
        # Load model, automatically handling LoRA adapters if present
        if os.path.exists(os.path.join(checkpoint_path, "adapter_config.json")):
            from peft import PeftModel
            with open(os.path.join(checkpoint_path, "adapter_config.json"), "r") as f:
                adapter_cfg = json.load(f)
            base_model_id = adapter_cfg.get("base_model_name_or_path", "Qwen/Qwen2.5-VL-7B-Instruct")
            logger.info(f"Loading base model {base_model_id}...")
            base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                base_model_id,
                torch_dtype=torch_dtype,
                device_map="auto" if self.device.type == "cuda" else None,
            )
            logger.info(f"Loading LoRA adapter from {checkpoint_path}...")
            self.model = PeftModel.from_pretrained(base_model, checkpoint_path)
        else:
            logger.info(f"Loading full model from {checkpoint_path}...")
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                checkpoint_path,
                torch_dtype=torch_dtype,
                device_map="auto" if self.device.type == "cuda" else None,
            )
        
        self.model.eval()

    def predict_next(self, i_final_path: str, prev_depth_path: str) -> str:
        """Predict instruction JSON text for next modeling step."""
        from qwen_vl_utils import process_vision_info
        
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": i_final_path},
                    {"type": "image", "image": prev_depth_path},
                    {"type": "text", "text": FIXED_USER_PROMPT},
                ]
            }
        ]
        
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs if video_inputs else None,
            padding=True,
            return_tensors="pt",
        ).to(self.device)
        
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=False,
            )
            
        generated_trimmed = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        
        output_text = self.processor.batch_decode(
            generated_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
        
        return output_text.strip()


def run_sequence_simulation(
    data_root: str,
    part_id: str,
    planner: QwenMLLMPlanner,
    output_dir: str
) -> None:
    """Simulates the step-by-step CAD reverse modeling sequence using dataset ground-truth transitions."""
    # Find part directory under data_root (could end with any suffix like _PPP, _NNN, etc.)
    part_dir = None
    for name in os.listdir(data_root):
        if name.startswith(part_id + "_") and os.path.isdir(os.path.join(data_root, name)):
            part_dir = os.path.join(data_root, name)
            break
            
    if part_dir is None:
        raise FileNotFoundError(f"Could not find part directory for '{part_id}' in '{data_root}'")
        
    final_snapshot_path = os.path.join(part_dir, "final_snapshot.png")
    if not os.path.isfile(final_snapshot_path):
        raise FileNotFoundError(f"Missing final_snapshot.png in {part_dir}")
        
    # Discover all step directories and sort them
    step_prefix = "roll_back_index_"
    step_indices = []
    for entry in os.listdir(part_dir):
        if entry.startswith(step_prefix) and os.path.isdir(os.path.join(part_dir, entry)):
            idx = int(entry[len(step_prefix):])
            step_indices.append(idx)
    step_indices.sort()
    
    if not step_indices:
        logger.warning(f"No step rollback directories found in {part_dir}")
        return
        
    logger.info(f"Starting simulation for part {part_id} with steps: {step_indices}")
    part_out_dir = os.path.join(output_dir, part_id)
    os.makedirs(part_out_dir, exist_ok=True)
    
    predictions = []
    
    # Iterate through each modeling step transition
    for step_idx in step_indices:
        step_dir = os.path.join(part_dir, f"{step_prefix}{step_idx}")
        prev_depth_path = os.path.join(step_dir, "prev_depth_map.png")
        gt_instruction_path = os.path.join(step_dir, "instruction.json")
        
        if not os.path.isfile(prev_depth_path):
            logger.warning(f"Missing prev_depth_map.png for step {step_idx}, skipping.")
            continue
            
        logger.info(f"--- Simulating Step {step_idx} ---")
        
        # Predict instruction JSON text
        raw_pred = planner.predict_next(final_snapshot_path, prev_depth_path)
        pred_dict = extract_json(raw_pred)
        
        # Save prediction
        out_pred_path = os.path.join(part_out_dir, f"step_{step_idx}_prediction.json")
        with open(out_pred_path, "w", encoding="utf-8") as f:
            json.dump({
                "raw_generation": raw_pred,
                "parsed": pred_dict
            }, f, indent=4)
            
        # Log comparison with ground-truth if available
        gt_dict = {}
        if os.path.isfile(gt_instruction_path):
            with open(gt_instruction_path, "r", encoding="utf-8") as f:
                gt_dict = json.load(f)
                
        logger.info(f"Predicted Operation: {pred_dict.get('operation_type')}")
        logger.info(f"Ground-Truth Operation: {gt_dict.get('operation_type')}")
        
        predictions.append({
            "step_index": step_idx,
            "ground_truth": gt_dict,
            "prediction": pred_dict
        })
        
    logger.info(f"Simulation completed. Saved predictions under: {part_out_dir}")


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    
    p = argparse.ArgumentParser(description="Phase 2 Autoregressive Reverse Modeling Planner Simulation.")
    p.add_argument("--checkpoint", type=str, required=True, help="Path to fine-tuned Qwen model folder.")
    p.add_argument("--data-root", type=str, default=None, help="Root folder containing parts dataset.")
    p.add_argument("--part-id", type=str, default=None, help="CAD part ID to run sequence simulation on.")
    p.add_argument("--i-final", type=str, default=None, help="Single snapshot path for individual prompt run.")
    p.add_argument("--depth-0", type=str, default=None, help="Single depth state path for individual prompt run.")
    p.add_argument("--output-dir", type=str, default="./generated")
    
    args = p.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    planner = QwenMLLMPlanner(args.checkpoint, device=device)
    
    # Mode 1: Sequence-level simulation on part dataset
    if args.data_root is not None and args.part_id is not None:
        run_sequence_simulation(
            data_root=args.data_root,
            part_id=args.part_id,
            planner=planner,
            output_dir=args.output_dir
        )
    # Mode 2: Single-step inference from individual snapshot and depth map
    elif args.i-final is not None and args.depth-0 is not None:
        logger.info("Running single-step planning prediction...")
        raw_pred = planner.predict_next(args.i_final, args.depth_0)
        parsed = extract_json(raw_pred)
        print("\n--- Raw Prediction ---")
        print(raw_pred)
        print("\n--- Parsed JSON ---")
        print(json.dumps(parsed, indent=4))
    else:
        logger.error(
            "Invalid arguments. Either specify (--data-root AND --part-id) for sequence simulation "
            "or (--i-final AND --depth-0) for single-step inference."
        )
        sys.exit(1)


if __name__ == "__main__":
    main()