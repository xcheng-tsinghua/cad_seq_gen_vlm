---
trigger: always_on
---

CAD Reverse Modeling Dataset Specification

1. Overview
This dataset is designed to train the "Vision-Based CAD Modeling Step Reverse Generation System". The dataset records every step of the CAD feature modeling process, from a blank slate to the final part. The current phase primarily validates single-view rendering and generation, but the file system retains an extensible structure for multi-view data.

2. Directory Structure
root
 ├─ [CAD_PART_ID]_[VIEW_SUFFIX] (e.g., PPP)
 │   ├─ roll_back_index_1 
 │   │   ├─ prev_depth_map.png 
 │   │   ├─ operation_param.json
 │   │   └─ overlayed_all.png
 │   │
 │   ├─ roll_back_index_3  # Note: indices might not be strictly continuous (e.g., jumps from 1 to 3)
 │   │   └─ ...
 │   │
 │   ├─ ...
 │   │
 │   └─ final_snapshot.png
 │
 ├─ [CAD_PART_ID]_[VIEW_SUFFIX] (e.g., NNN)
 │   ├─ roll_back_index_1
 │   │   └─ ...
 │   └─ ...
 ...

3. File Definitions
3.1 Sequence-Level Files
Located in the root directory of the [CAD_PART_ID]_[VIEW_SUFFIX] folder, representing the final global state of the part.

  final_snapshot.png: The color rendered image of the final part from this specific view. Purpose: Serves as the Global Context, fed into the MLLM to help it understand the macroscopic location of local features; also used as the reference input for the Diffusion model's IP-Adapter.

3.2 Step-Level Files
Located within the roll_back_index_{i} folders, representing the state changes at the i-th operational step.

  prev_depth_map.png: The depth map (grayscale) of the existing model before the current modeling operation is executed. Purpose: Serves as the spatial condition input for the ControlNet in the Diffusion model, providing a "clean geometric base".

  operation_param.json: The ground-truth parameters of the current modeling step (including operation type modeling_type, depth, construct_type, etc.). Purpose: Provides absolutely accurate underlying logic for the MLLM to generate training prompts, eliminating visual ambiguity.

  overlayed_all.png: (Core File) The 4-in-1 Alpha blended composite image. A human-vision-friendly image created by overlaying the base depth map, sketch plane mask, reference mask, and colored incremental wireframe. Purpose: Serves as the "local context" image for the MLLM to understand the current operational step, and also functions as the Target image for the Diffusion model's image generation.

4. Global Core Rules & Visual Anchors
Rule A: Color Mask Encoding
  In overlayed_all.png, colors represent unique topological, geometric, and physical meanings:

  Semi-Transparent Yellow masked area: Sketch Plane.

  Semi-Transparent Cyan masked area: Reference Geometry (e.g., Path / Axis of Revolution).

  Red (Solid Line): Reference 2D Sketch used in the current operation.

  Green (Solid Line): Edges of the newly added solid entity in this step (Added Solid Entity / construct_type: "NEW", "ADD", ...).

  Magenta (Solid Line): Edges of the entity cut or removed in this step (Removed Entity / construct_type: "CUT", "REMOVE", ...).

  Blue (Solid Line): The termination face contour of the extrusion/sweep in this step (Termination Face).
