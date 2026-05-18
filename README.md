
# environment
这里 cuda 版本是 12.2
conda create -n cad_vlm python=3.11.11
conda activate cad_vlm
pip install torch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0 --index-url https://download.pytorch.org/whl/cu121

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
- **分辨率策略**：读取时使用等比缩放 + padding，避免拉伸变形；推理导出时自动去 padding 并恢复原图比例。
- **专门 loss**：
  - `mask IoU`（`sketch_plane_mask` / `reference_mask`）
  - `wireframe edge consistency`（`result_frame` 的 Sobel 梯度一致性）
  - `SD latent consistency`（使用最新 Stable Diffusion 的 VAE 做结构感知约束）
- **步骤数预测**：CLIP + KNN

## 最新 Stable Diffusion

训练脚本固定使用以下路径：

- `vlm/stable-diffusion-3.5-medium/sd3.5_medium.safetensors`
- `vlm/stable-diffusion-3.5-medium/sd3.5_medium_config.json`

如果任一文件不存在，会从 HuggingFace 下载后复制到上述固定位置。
其中：
- `sd3.5_medium.safetensors`：下载 `sd3.5_medium.safetensors`
- `sd3.5_medium_config.json`：下载 `vae/config.json` 并重命名保存

## HuggingFace 登录

`stabilityai/stable-diffusion-3.5-medium` 是 gated 模型，下载前需要登录。  
Access Token 固定从 `vlm/hf_access_token.json` 读取。

```bash
# 请直接编辑文件：
# vlm/hf_access_token.json
# {
#   "access_token": "hf_xxx"
# }
```

如果 `--w-sd-latent 0` 则不启用 VAE，跳过登录与下载。

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
python -m train --raw-root "E:/your_dataset_root"
```

输出：
- `model_trained/<dataset_name>/train_<timestamp>/best.pth`
- `model_trained/<dataset_name>/train_<timestamp>/last.pth`
- `model_trained/<dataset_name>/train_<timestamp>/train_history.json`

### 3) 推理（自回归步骤生成）

```bash
python -m infer --raw-root "E:/your_dataset_root" --input-image "E:/test/part.png"
```

默认会自动寻找最近一次训练的 `best.pth`，并把结果保存到 `output/<dataset_name>/infer_<timestamp>/`。

### 4) 验证与可视化评估

```bash
python -m eval --raw-root "E:/your_dataset_root"
```

输出：
- `metrics.json`（四个子图分开评估）
- `summary_metrics.png`（汇总柱状图）
- `visuals/*.png`（预测 vs GT 对比图）

评估结果保存到 `output/<dataset_name>/eval_<timestamp>/`。

