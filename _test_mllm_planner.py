"""Verification tests for the SFT Dataset, Tokenizer tracking, Differentiable coordinate decoding, and Gaussian rendering."""

import os
import sys
import json
import tempfile
import numpy as np
import torch
from PIL import Image

sys.path.insert(0, ".")

from mllm_planner_dataset import MLLMPlannerSFTDataset, collate_planner_batch
from train_mllm_planner import (
    decode_expectation_coordinates_vectorized,
    render_heatmap,
    map_spans_to_tokens,
    find_sublist,
    CONTOUR_KEY_MAPPING
)


def write_image(path, arr, mode="L"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    Image.fromarray(arr.astype("uint8"), mode).save(path)


def test_dataset_and_collate():
    print("Testing MLLMPlannerSFTDataset and collation...")
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a mock part
        part_name = "PART_TEST_PPP"
        part_dir = os.path.join(tmpdir, part_name)
        os.makedirs(part_dir, exist_ok=True)
        
        # Write final snapshot
        write_image(
            os.path.join(part_dir, "final_snapshot.png"),
            np.full((128, 128, 3), 100, dtype=np.uint8),
            mode="RGB"
        )
        
        # Write step 1
        step_dir = os.path.join(part_dir, "roll_back_index_1")
        os.makedirs(step_dir, exist_ok=True)
        write_image(os.path.join(step_dir, "prev_depth_map.png"), np.full((64, 64), 50, dtype=np.uint8))
        
        # Write mock instruction.json
        mock_instruction = {
            "operation_type": "extrude_add",
            "sketch (red)": [[10, 20], [30, 40]],
            "other_counter (green / magenta)": [[50, 60]],
            "terminate_face_contour (blue)": [],
            "sketch_plane_contour (yellow)": [[12, 34]],
            "reference_geom_contour (cyan)": [],
            "target_region_bbox (red)": [5, 15, 25, 35]
        }
        with open(os.path.join(step_dir, "instruction.json"), "w", encoding="utf-8") as f:
            json.dump(mock_instruction, f)
            
        # Write mock contour PNGs
        write_image(os.path.join(step_dir, "sketch.png"), np.zeros((100, 100)))
        write_image(os.path.join(step_dir, "other_counter.png"), np.zeros((100, 100)))
        write_image(os.path.join(step_dir, "sketch_plane_contour.png"), np.zeros((100, 100)))
        
        # Load dataset
        ds = MLLMPlannerSFTDataset(data_root=tmpdir, contour_size=(100, 100))
        assert len(ds) == 1
        
        item = ds[0]
        assert item["part_id"] == "PART_TEST"
        assert item["step_index"] == 1
        assert "instruction_text" in item
        assert "spans" in item
        
        # Check quantized coordinates in item
        inst = item["instruction"]
        assert inst["operation_type"] == "extrude_add"
        assert inst["sketch (red)"] == [[10, 20], [30, 40]]
        assert inst["target_region_bbox (red)"] == [5, 15, 25, 35]
        
        # Check contours shape
        for c_key in CONTOUR_KEY_MAPPING.keys():
            assert c_key in item["contours"]
            assert item["contours"][c_key].shape == (1, 100, 100)
            
        # Test collate
        batch = collate_planner_batch([item])
        assert len(batch["part_id"]) == 1
        assert batch["contours"]["sketch"].shape == (1, 1, 100, 100)
        
    print("Dataset & Collate Tests Passed!")


def test_span_mapping():
    print("Testing span and sublist mapping...")
    # A simple mock text
    # target text
    target_text = '{\n    "sketch (red)": [[12,34], [56,78]]\n}'
    spans = {
        "sketch (red)": [(21, 23), (24, 26), (29, 31), (32, 34)]
    }
    
    # Mock a tokenizer behavior
    class MockTokenizer:
        def encode(self, text, **kwargs):
            # A dummy tokenization
            # Let's say it tokenizes characters
            return [ord(c) for c in text]
            
        def __call__(self, text, **kwargs):
            # return input_ids and offsets
            input_ids = [ord(c) for c in text]
            # offset mapping maps each character token to (i, i+1)
            offsets = [(i, i+1) for i in range(len(text))]
            return {"input_ids": input_ids, "offset_mapping": offsets}
            
    tokenizer = MockTokenizer()
    
    # Check find_sublist
    main_list = [1, 2, 3, 4, 5, 6]
    sub_list = [3, 4, 5]
    assert find_sublist(main_list, sub_list) == 2
    
    # Map spans to tokens
    offsets = [(i, i+1) for i in range(len(target_text))]
    token_indices = map_spans_to_tokens(offsets, spans)
    
    # 12 is at char 21-23. Since offset mapping is char by char:
    # Char 21 matches ord('1') (idx 21), Char 22 matches ord('2') (idx 22)
    # The map_spans_to_tokens helper searches for token that overlaps
    sketch_indices = token_indices["sketch (red)"]
    assert len(sketch_indices) == 4
    for idx in sketch_indices:
        assert idx != -1
        
    print("Span mapping tests passed!")


def test_differentiable_gradients():
    print("Testing differentiable coordinate decoding and Gaussian rendering backprop...")
    device = torch.device("cpu")
    
    # Parameters
    vocab_size = 1000
    seq_len = 50
    batch_size = 1
    
    # Mock logits
    logits = torch.randn(batch_size, seq_len, vocab_size, requires_grad=True)
    
    # Precompute mock number token IDs: let "0"-"100" be mapped to token IDs 500-600
    number_token_ids = torch.arange(500, 601, dtype=torch.long, device=device)
    
    # Coordinate index positions (corresponding to sketch (red) x1, y1)
    indices_list = [10, 11] # x1 token index = 10, y1 token index = 11
    
    # Decode expectation coordinates
    pred_points = decode_expectation_coordinates_vectorized(
        logits[0], indices_list, number_token_ids, device
    ) # shape: (N, 2)
    
    assert pred_points.shape == (1, 2)
    
    # Render heatmap
    H, W = 100, 100
    sigma = 2.0
    pred_heatmap = render_heatmap(pred_points, H, W, sigma, device, logits.dtype)
    
    assert pred_heatmap.shape == (1, 100, 100)
    
    # Ground truth heatmap: a point at (10, 20)
    gt_heatmap = torch.zeros(1, 100, 100, dtype=logits.dtype)
    # Render a ground-truth point at x=10, y=20 (scaled to grid index)
    gt_x = int(round(10.0 * 99 / 100.0))
    gt_y = int(round(20.0 * 99 / 100.0))
    gt_heatmap[0, gt_y, gt_x] = 1.0
    
    # Compute MSE loss
    loss = torch.mean((pred_heatmap - gt_heatmap) ** 2)
    
    # Backprop
    loss.backward()
    
    # Verify that logits.grad is NOT None and contains non-zero values
    assert logits.grad is not None
    
    # Gradients should only exist at indices_list - 1 (causal shift)
    # e.g., index 9 and 10
    grad_at_x = logits.grad[0, 9, number_token_ids]
    grad_at_y = logits.grad[0, 10, number_token_ids]
    
    assert torch.any(grad_at_x != 0)
    assert torch.any(grad_at_y != 0)
    
    print("Gradients flow verified! Gradients at causal prediction steps are non-zero.")
    print("Differentiable rendering backpropagation successfully verified.")


def main():
    test_dataset_and_collate()
    test_span_mapping()
    test_differentiable_gradients()
    print("\nALL VERIFICATION TESTS PASSED SUCCESSFULLY!")


if __name__ == "__main__":
    main()
