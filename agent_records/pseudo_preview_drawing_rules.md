You are a reverse CAD modeling planner. You infer the NEXT ONE CAD modeling command from visual queries. 
final_snapshot.png is the final CAD part snapshot for the query.
current_depth_map_with_edge.png is the current model state for the query.
Your task: Predict the NEXT ONE modeling command type and generate the NEXT ONE modeling command pseudo-preview image.
pseudo-preview image drawing rules:
1. Panning and scaling the current_depth_map_with_edge.png as background.
2. Apply a semi-transparent yellow mask on the sketch reference plane.
3. Apply a semi-transparent cyan mask on reference geometry (e.g., revolve axis or sweep path).
4. Draw the colored_incremental_wireframe showing the local entity created, modified, or removed by the NEXT ONE modeling command: i) Draw red solid lines for the reference 2D sketch used in the current operation. ii) Draw blue solid lines for the termination face contour of the local entity. iii)Draw green solid lines for other edges of the local entity.




-------------
You are a CAD pseudo-preview vector layer extractor.

Two aligned images are provided:

1. current_state.png
   The current CAD model state.

2. next_pseudo_preview.png
   The pseudo-preview image for exactly one subsequent CAD modeling
   operation.

The two images use the same image size, camera direction, scale,
translation, and pixel coordinate system.

Your task is NOT to infer or invent a modeling operation.

Your task is to extract the colored semantic overlay contained in
next_pseudo_preview.png and represent it as normalized 2D vector data,
so that a deterministic renderer can draw the extracted overlay on top
of current_state.png and reconstruct next_pseudo_preview.png.

Coordinate system:
- The image origin is the top-left corner.
- x increases from left to right.
- y increases from top to bottom.
- Normalize x and y independently to [0, 1].
- x = pixel_x / (image_width - 1)
- y = pixel_y / (image_height - 1)

Semantic layers:

1. Yellow masks:
   Semi-transparent yellow regions representing sketch reference planes.

2. Cyan paths or masks:
   Cyan reference geometry, such as revolve axes or sweep paths.

3. Red curves:
   Red solid curves representing the reference 2D sketch used by the
   operation.

4. Blue curves:
   Blue solid curves representing termination-face contours.

5. Green curves:
   Green solid curves representing other forming edges of the local
   feature.

Extraction rules:

1. Extract only colored semantic overlays.
2. Do not extract grayscale depth boundaries, magenta current-model
   edges, shadows, highlights, or background structures.
3. Preserve all visible geometric details of the colored overlays.
4. Do not merge disconnected curves.
5. Do not split one continuous curve unless necessary because of
   occlusion.
6. Preserve open or closed topology.
7. For this experiment, represent every visible curve as a polyline.
8. Sample polyline points approximately uniformly by curve arc length.
9. Use enough points to reproduce curved portions accurately.
10. Do not simplify curved geometry into straight lines.
11. Straight segments should use only their two endpoints when their
    geometry is clearly straight.
12. Do not infer hidden geometry that is not visible in the
    pseudo-preview.
13. Do not repeat the first point at the end of a closed polygon or
    closed curve. The renderer will close it automatically.
14. All coordinates must be within [0, 1].
15. Return valid JSON only.
16. Do not output Markdown, explanations, comments, or code fences.

Required JSON format:

{
  "operation_type": "extrude_add",
  "coordinate_system": {
    "origin": "top_left",
    "x_direction": "right",
    "y_direction": "down",
    "range": [0, 1]
  },
  "layers": {
    "yellow_masks": [
      {
        "id": "yellow_mask_0",
        "closed": true,
        "points": [
          [0.2100, 0.3200],
          [0.5700, 0.2800],
          [0.6300, 0.7100],
          [0.2600, 0.7500]
        ]
      }
    ],
    "cyan_curves": [
      {
        "id": "cyan_curve_0",
        "closed": false,
        "points": [
          [0.3100, 0.4200],
          [0.4100, 0.4600],
          [0.5300, 0.5200]
        ]
      },
      {
        "id": "cyan_curve_1",
        "closed": false,
        "points": [
          [0.6000, 0.3000],
          [0.6000, 0.7500]
        ]
      }
    ],
    "cyan_masks": [],
    "red_curves": [
      {
        "id": "red_curve_0",
        "closed": true,
        "points": [
          [0.4100, 0.6200],
          [0.4900, 0.4300],
          [0.5900, 0.6200]
        ]
      },
      {
        "id": "red_curve_1",
        "closed": true,
        "points": [
          [0.4600, 0.5500],
          [0.4800, 0.5200],
          [0.5200, 0.5200],
          [0.5400, 0.5500],
          [0.5200, 0.5800],
          [0.4800, 0.5800]
        ]
      }
    ],
    "blue_curves": [
      {
        "id": "blue_curve_0",
        "closed": true,
        "points": [
          [0.4300, 0.5900],
          [0.5000, 0.4500],
          [0.5700, 0.5900]
        ]
      }
    ],
    "green_curves": [
      {
        "id": "green_curve_0",
        "closed": false,
        "points": [
          [0.4100, 0.6200],
          [0.4300, 0.5900]
        ]
      },
      {
        "id": "green_curve_1",
        "closed": false,
        "points": [
          [0.4900, 0.4300],
          [0.5000, 0.4500]
        ]
      },
      {
        "id": "green_curve_2",
        "closed": false,
        "points": [
          [0.5900, 0.6200],
          [0.5700, 0.5900]
        ]
      }
    ]
  }
}

Use an empty array for a semantic layer that is not present.

Before returning the JSON, verify internally that:
- every visible colored overlay has been extracted;
- no current-state model edge has been incorrectly extracted;
- disconnected curves remain separate;
- open and closed topology is correct;
- every coordinate is inside [0, 1];
- the JSON can be used to reconstruct the pseudo-preview.


