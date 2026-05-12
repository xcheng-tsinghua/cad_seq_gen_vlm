# CAD Sequence Generation (Python)

这个项目用于从用户输入的 CAD 零件图片生成一组建模步骤序列。  
每个建模步骤输出 4 张图（打包为一个 `step canvas`）：

1. `prev_depth_map`：执行当前步骤前的深度图
2. `sketch_plane_mask`：草图基准面 mask
3. `reference_mask`：参考几何体 mask
4. `result_frame`：当前步骤生成实体线框图

## 方案概述

- **模型骨干**：开源 `Stable Diffusion + ControlNet`
- **少样本微调**：UNet LoRA 微调（参数量小，适合你的数据规模）
- **序列生成**：自回归逐步生成 step canvas（每一步再拆分成 4 张图）
- **步数预测**：基于 CLIP 特征的 KNN 回归（轻量、可解释）

## 数据目录（原始）

项目假定你已有如下结构（与你提供的一致）：

```text
root/
  <part_id_1>/
    roll_back_index_1/
      prev_depth_map.png
      sketch_plane_mask.png
      reference_mask.png
      result_frame.png
    roll_back_index_3/
      ...
  <part_id_2>/
    ...
```

## 一键流程

### 1) 安装依赖

```bash
pip install -r requirements.txt
```

### 2) 预处理数据

```bash
python -m src.cad_seq_gen.data.prepare_dataset ^
  --raw-root "E:/your_dataset_root" ^
  --out-root "E:/your_processed_root" ^
  --image-size 512
```

输出内容：
- `manifest.jsonl`：训练样本索引（每行为一个步骤）
- `targets/*.png`：每个步骤的 2x2 目标拼图（四图合一）
- `controls/*.png`：对应控制图（目标零件图 + 上一步信息）
- `train_split.json` / `val_split.json`
- `step_stats.json`：每个 part 的步骤数统计

### 3) 微调 ControlNet + LoRA

```bash
python -m src.cad_seq_gen.train_controlnet_lora ^
  --processed-root "E:/your_processed_root" ^
  --pretrained-model "runwayml/stable-diffusion-v1-5" ^
  --controlnet-model "lllyasviel/sd-controlnet-canny" ^
  --output-dir "E:/outputs/cad_seq_lora" ^
  --epochs 20 ^
  --batch-size 2 ^
  --lr 1e-4
```

> 说明：`controlnet-model` 可以替换为你后续自行训练/更适配 CAD 的 ControlNet 初始化权重。

### 4) 推理建模序列

```bash
python -m src.cad_seq_gen.infer_sequence ^
  --input-image "E:/test/part.png" ^
  --processed-root "E:/your_processed_root" ^
  --pretrained-model "runwayml/stable-diffusion-v1-5" ^
  --controlnet-model "lllyasviel/sd-controlnet-canny" ^
  --lora-dir "E:/outputs/cad_seq_lora" ^
  --output-dir "E:/outputs/infer_case_001" ^
  --num-steps 0
```

`--num-steps 0` 表示自动预测步骤数；你也可以手工指定。

输出目录示例：

```text
infer_case_001/
  step_001/
    prev_depth_map.png
    sketch_plane_mask.png
    reference_mask.png
    result_frame.png
  step_002/
    ...
```

## 关键设计细节

- 用 `2x2 canvas` 表达单步四图，训练和推理统一。
- 每一步控制图包含：
  - 目标零件图（用户输入图）
  - 上一步生成结果（自回归上下文）
  - 辅助边缘先验（从目标零件图提取）
- 通过 CLIP KNN 回归预测步骤数，减少手工指定步骤数。

## 后续建议（你数据少时很重要）

- 先将 `image_size` 设为 `384` 或 `512`，优先验证闭环可用性。
- 用 LoRA + 冻结大部分参数，避免过拟合。
- 增强 mask（随机膨胀/腐蚀、轻微仿射、噪声）提升泛化。
- 评估时分开统计四个子图质量（IoU、Chamfer、线框重投影误差）。

