# CAD Sequence Generation (Structured V2 Only)

这个仓库只保留第二版：**结构化多头生成**。  
每个建模步骤同时预测 4 张子图（不是单一 canvas）：

1. `prev_depth_map`  
2. `sketch_plane_mask`  
3. `reference_mask`  
4. `result_frame`

## 模型设计

- **主模型**：`StructuredMultiHeadUNet`（共享编码器 + 4 个输出头）
- **输入通道**：
  - 目标零件图
  - 上一步 4 个子图
  - 目标零件边缘图
- **专门 loss**：
  - `mask IoU`（`sketch_plane_mask` / `reference_mask`）
  - `wireframe edge consistency`（`result_frame` 的 Sobel 梯度一致性）
  - `SD latent consistency`（使用最新 Stable Diffusion 的 VAE 做结构感知约束）
- **步骤数预测**：CLIP + KNN

## 最新 Stable Diffusion

训练默认使用：

- `stabilityai/stable-diffusion-3.5-medium`

作为 VAE 感知约束来源（`--sd-model-id` 可改）。

## 原始数据目录

```text
root/
  <part_id_1>/
    roll_back_index_1/
      prev_depth_map.png
      sketch_plane_mask.png
      reference_mask.png
      result_frame.png
    ...
```

## 使用流程

### 1) 安装依赖

```bash
pip install -r requirements.txt
```

### 2) 训练（结构化多头，在线预处理）

```bash
python -m src.cad_seq_gen.train ^
  --raw-root "E:/your_dataset_root" ^
  --output-dir "E:/outputs/structured_v2" ^
  --image-size 384 ^
  --epochs 80 ^
  --batch-size 8 ^
  --sd-model-id "stabilityai/stable-diffusion-3.5-medium" ^
  --w-sd-latent 0.2
```

输出：
- `best.pt`
- `last.pt`
- `train_history.json`

### 3) 推理（自回归步骤生成）

```bash
python -m src.cad_seq_gen.infer ^
  --input-image "E:/test/part.png" ^
  --raw-root "E:/your_dataset_root" ^
  --checkpoint "E:/outputs/structured_v2/best.pt" ^
  --output-dir "E:/outputs/infer_case_001" ^
  --num-steps 0
```

`--num-steps 0` 表示自动预测步数。

### 4) 验证与可视化评估

```bash
python -m src.cad_seq_gen.eval ^
  --raw-root "E:/your_dataset_root" ^
  --checkpoint "E:/outputs/structured_v2/best.pt" ^
  --output-dir "E:/outputs/eval_structured"
```

输出：
- `metrics.json`（四个子图分开评估）
- `summary_metrics.png`（汇总柱状图）
- `visuals/*.png`（预测 vs GT 对比图）

## PowerShell 快捷脚本

- `scripts/train.ps1`
- `scripts/infer.ps1`
- `scripts/eval.ps1`

推荐直接用脚本，路径自动管理：

```powershell
# 只需要给数据集路径
powershell -ExecutionPolicy Bypass -File .\scripts\train.ps1 -RawRoot "E:\your_dataset_root"

# 推理只需要数据集路径 + 输入零件图（checkpoint 自动找最近一次训练）
powershell -ExecutionPolicy Bypass -File .\scripts\infer.ps1 -RawRoot "E:\your_dataset_root" -InputImage "E:\test\part.png"

# 评估只需要数据集路径（checkpoint 自动找最近一次训练）
powershell -ExecutionPolicy Bypass -File .\scripts\eval.ps1 -RawRoot "E:\your_dataset_root"
```

输出会自动保存到：

- `outputs/<dataset_name>/train_<timestamp>/`
- `outputs/<dataset_name>/infer_<timestamp>/`
- `outputs/<dataset_name>/eval_<timestamp>/`

