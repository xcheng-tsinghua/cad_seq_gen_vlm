"""Phase 1 — Qwen2.5-VL planner fine-tuning (SFT with differentiable coordinate loss)."""

from __future__ import annotations

import argparse
import logging
import os
import sys
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from PIL import Image

from mllm_planner_dataset import MLLMPlannerSFTDataset, collate_planner_batch

# Define the user prompt exactly as required by the reverse modeling system
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

CONTOUR_KEY_MAPPING = {
    "sketch": "sketch (red)",
    "other_counter": "other_counter (green / magenta)",
    "terminate_face_contour": "terminate_face_contour (blue)",
    "sketch_plane_contour": "sketch_plane_contour (yellow)",
    "reference_geom_contour": "reference_geom_contour (cyan)",
}

logger = logging.getLogger(__name__)


def find_sublist(main: List[int], sub: List[int]) -> int:
    """Find the start index of sublist in main list."""
    n = len(main)
    m = len(sub)
    for i in range(n - m + 1):
        if main[i : i + m] == sub:
            return i
    return -1


def map_spans_to_tokens(offset_mapping: List[Tuple[int, int]], spans: Dict[str, List[Tuple[int, int]]]) -> Dict[str, List[int]]:
    """Map character spans of coordinate numbers to token indices."""
    token_indices = {}
    for key, span_list in spans.items():
        indices = []
        for start_char, end_char in span_list:
            found = False
            for t_idx, (t_start, t_end) in enumerate(offset_mapping):
                if t_start == 0 and t_end == 0:
                    continue  # skip special tokens
                if t_start <= start_char and t_end >= end_char:
                    indices.append(t_idx)
                    found = True
                    break
            if not found:
                for t_idx, (t_start, t_end) in enumerate(offset_mapping):
                    if t_start == 0 and t_end == 0:
                        continue
                    if not (t_end <= start_char or t_start >= end_char):
                        indices.append(t_idx)
                        found = True
                        break
            if not found:
                indices.append(-1)
        token_indices[key] = indices
    return token_indices


def decode_expectation_coordinates_vectorized(
    logits: torch.Tensor,
    indices_list: List[int],
    number_token_ids: torch.Tensor,
    device: torch.device
) -> torch.Tensor:
    """Compute soft expected coordinate values over token logits."""
    if len(indices_list) < 2:
        return torch.empty(0, 2, dtype=logits.dtype, device=device)
    
    x_toks = torch.tensor(indices_list[0::2], dtype=torch.long, device=device)
    y_toks = torch.tensor(indices_list[1::2], dtype=torch.long, device=device)
    
    # Causal shift: logits predicting token at t_idx are at t_idx - 1
    x_logits = logits[x_toks - 1][:, number_token_ids]
    y_logits = logits[y_toks - 1][:, number_token_ids]
    
    x_probs = torch.softmax(x_logits, dim=-1)
    y_probs = torch.softmax(y_logits, dim=-1)
    
    arange_101 = torch.arange(101, dtype=logits.dtype, device=device)
    
    x_vals = torch.sum(x_probs * arange_101, dim=-1)
    y_vals = torch.sum(y_probs * arange_101, dim=-1)
    
    return torch.stack([x_vals, y_vals], dim=-1)


def render_heatmap(
    predicted_points: torch.Tensor,
    H: int,
    W: int,
    sigma: float,
    device: torch.device,
    dtype: torch.dtype
) -> torch.Tensor:
    """Differentiable Gaussian Renderer for points to a 2D heatmap/pixel map."""
    N = predicted_points.shape[0]
    if N == 0:
        return torch.zeros(1, H, W, dtype=dtype, device=device)
    
    # Scale from [0, 100] to grid boundaries
    x_grid = (predicted_points[:, 0] * ((W - 1) / 100.0)).view(-1, 1, 1)
    y_grid = (predicted_points[:, 1] * ((H - 1) / 100.0)).view(-1, 1, 1)
    
    y_indices = torch.arange(H, dtype=dtype, device=device).view(H, 1).expand(H, W)
    x_indices = torch.arange(W, dtype=dtype, device=device).view(1, W).expand(H, W)
    
    x_ind = x_indices.unsqueeze(0)
    y_ind = y_indices.unsqueeze(0)
    
    dist_sq = (x_ind - x_grid) ** 2 + (y_ind - y_grid) ** 2
    intensities = torch.exp(-dist_sq / (2.0 * (sigma ** 2)))
    
    sum_intensities = torch.sum(intensities, dim=0, keepdim=True)
    heatmap = torch.clamp(sum_intensities, 0.0, 1.0)
    return heatmap


def preprocess_batch(
    batch: Dict[str, object],
    processor: object,
    device: torch.device
) -> Tuple[Dict[str, torch.Tensor], List[Dict[str, List[int]]]]:
    """Helper to preprocess training messages and align tokens with coordinate indices."""
    from qwen_vl_utils import process_vision_info
    
    batch_size = len(batch["part_id"])
    batch_messages = []
    
    for i in range(batch_size):
        msgs = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": batch["final_snapshot_path"][i]},
                    {"type": "image", "image": batch["prev_depth_path"][i]},
                    {"type": "text", "text": FIXED_USER_PROMPT},
                ]
            },
            {
                "role": "assistant",
                "content": batch["instruction_text"][i]
            }
        ]
        batch_messages.append(msgs)
        
    texts = [processor.apply_chat_template(m, tokenize=False) for m in batch_messages]
    
    image_inputs = []
    video_inputs = []
    for m in batch_messages:
        imgs, vids = process_vision_info(m)
        image_inputs.extend(imgs)
        video_inputs.extend(vids)
        
    inputs = processor(
        text=texts,
        images=image_inputs,
        videos=video_inputs if video_inputs else None,
        padding=True,
        return_tensors="pt",
    ).to(device)
    
    labels = torch.full_like(inputs["input_ids"], -100)
    batch_coord_indices = []
    
    for i in range(batch_size):
        target_text = batch["instruction_text"][i]
        spans = batch["spans"][i]
        
        target_encoding = processor.tokenizer(
            target_text,
            return_offsets_mapping=True,
            add_special_tokens=False,
        )
        target_ids = target_encoding["input_ids"]
        target_offsets = target_encoding["offset_mapping"]
        
        target_coord_tok_indices = map_spans_to_tokens(target_offsets, spans)
        
        input_ids_list = inputs["input_ids"][i].tolist()
        start_idx = find_sublist(input_ids_list, target_ids)
        
        if start_idx != -1:
            labels[i, start_idx : start_idx + len(target_ids)] = inputs["input_ids"][i, start_idx : start_idx + len(target_ids)]
            
            abs_indices = {}
            for key, indices in target_coord_tok_indices.items():
                abs_indices[key] = [idx + start_idx for idx in indices if idx != -1]
            batch_coord_indices.append(abs_indices)
        else:
            batch_coord_indices.append({k: [] for k in spans.keys()})
            
    inputs["labels"] = labels
    return inputs, batch_coord_indices


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase 1 — Qwen planner SFT with differentiable coordinate loss.")
    p.add_argument("--data-root", type=str, required=True)
    p.add_argument("--part-ids-file", type=str, default=None)
    p.add_argument("--output-dir", type=str, default="./checkpoints")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--sigma", type=float, default=2.0)
    p.add_argument("--lambda-coord", type=float, default=10.0)
    p.add_argument("--model-id", type=str, default="Qwen/Qwen2.5-VL-7B-Instruct")
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    return p.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    args = _parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load dataset
    ds = MLLMPlannerSFTDataset(
        data_root=args.data_root,
        contour_size=(100, 100),
        part_ids_file=args.part_ids_file,
    )
    
    dl = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_planner_batch,
        drop_last=True
    )
    
    # Load models
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    from peft import LoraConfig, get_peft_model
    
    processor = AutoProcessor.from_pretrained(args.model_id)
    
    # Load model in bfloat16 if CUDA is available, else float32
    torch_dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_id,
        torch_dtype=torch_dtype,
        device_map="auto" if device.type == "cuda" else None,
    )
    
    # PEFT LoRA Config
    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    
    # Precompute token IDs for number strings "0" to "100"
    number_token_ids = []
    for i in range(101):
        ids = processor.tokenizer.encode(str(i), add_special_tokens=False)
        number_token_ids.append(ids[0])
    number_token_ids = torch.tensor(number_token_ids, dtype=torch.long, device=device)
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    
    model.train()
    for epoch in range(args.epochs):
        epoch_loss = 0.0
        pbar = tqdm(dl, desc=f"Epoch {epoch + 1}/{args.epochs}")
        for batch in pbar:
            optimizer.zero_grad()
            
            # Preprocess inputs, labels, and track coords
            inputs, batch_coord_indices = preprocess_batch(batch, processor, device)
            
            # Causal LM Forward
            outputs = model(**inputs)
            sft_loss = outputs.loss
            logits = outputs.logits # (batch, seq_len, vocab)
            
            # Coordinate pixel mask loss
            coord_loss = 0.0
            batch_size = len(batch["part_id"])
            
            for i in range(batch_size):
                item_logits = logits[i]
                coord_indices = batch_coord_indices[i]
                
                for c_key, span_key in CONTOUR_KEY_MAPPING.items():
                    indices_list = coord_indices.get(span_key, [])
                    gt_heatmap = batch["contours"][c_key][i].to(device=device, dtype=item_logits.dtype)
                    H, W = gt_heatmap.shape[1], gt_heatmap.shape[2]
                    
                    pred_points = decode_expectation_coordinates_vectorized(
                        item_logits, indices_list, number_token_ids, device
                    )
                    
                    pred_heatmap = render_heatmap(
                        pred_points, H, W, args.sigma, device, item_logits.dtype
                    )
                    
                    loss_val = torch.mean((pred_heatmap - gt_heatmap) ** 2)
                    coord_loss = coord_loss + loss_val
            
            # Combine losses
            loss = sft_loss + args.lambda_coord * (coord_loss / batch_size)
            
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "sft": f"{sft_loss.item():.4f}",
                "coord": f"{(coord_loss.item() / batch_size):.4f}"
            })
            
        avg_loss = epoch_loss / len(dl)
        logger.info(f"Epoch {epoch + 1} finished, Average Loss: {avg_loss:.4f}")
        
        # Save check points
        checkpoint_dir = os.path.join(args.output_dir, f"epoch_{epoch + 1}")
        model.save_pretrained(checkpoint_dir)
        processor.save_pretrained(checkpoint_dir)
        logger.info(f"Saved checkpoint to {checkpoint_dir}")


if __name__ == "__main__":
    main()
