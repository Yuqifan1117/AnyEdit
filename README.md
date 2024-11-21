# AnyEdit
Mastering Unified High-Quality Image Editing for Any Idea
# 🌍 Introduction
AnyEdit is a comprehensive multimodal instruction editing dataset, comprising 2.5 million high-quality editing pairs spanning over 20 editing types across five domains. We ensure the diversity and quality of the AnyEdit collection through three aspects: initial data diversity, adaptive editing process, and automated selection of editing results. Using the dataset, we further train a novel AnyEdit Stable Diffusion with task-aware routing and learnable task embedding for unified image editing. Comprehensive experiments on three benchmark datasets show that AnyEdit consistently boosts the performance of diffusion-based editing models. This presents prospects for developing instruction-driven image editing models that support human creativity.

# 💡 Overview
![image](figures/main_figures/figure1_dataset_00.png)
We comprehensively categorize image editing tasks into 5 groups based on different editing capabilities:
- (a) Local Editing which focuses on region-based editing (green area);
- (b) Global Editing which focuses on the full range of image rendering (yellow area);
- (c) Camera Move Editing which focuses on viewpoints changing instead of scenes (gray area);
- (d) Implicit Editing which requires commonsense knowledge to complete complex editing (orange area);
- (e) Visual Editing which encompasses additional visual inputs, addressing the requirements for multi-modal editing (blue area).

# ⭐ Steps for AnyEdit Collection 
![image](figures/main_figures/pipeline_00.png)
1. General Data Preparation
2. Diverse Instruction Generation
3. Adaptive Editing Pipelines
4. Data Quality Enhancement
  ### Instruction Format
  ```python
  {
    "edit": "change the airplane to green",  # edited instruction
    "edited object": "airplane",   # the edited region, only for local editing, else is None
    "input": "a small airplane sits stationary on a piece of concrete.",  # the caption of the original image 
    "output": "A green small airplane sits stationary on a piece of concrete.",  # the caption of the edited image 
    "edit_type": "color_alter",  # editing type
    "visual_input": "None", # the reference image for visual input instruction, else is None
    "image_file": "COCO_train2014_000000521165.jpg", # the file of original image
    "edited_file": "xxxxx.png"  # the file of edited image
  }
  ```

  ### Instruciton Pipeline
  ![image](figures/main_figures/pipelines.png)

# 🛠️ Setups for AnyEdit
0. Conda a new python environment and Download the pretrained weights

```bash
bash setup.sh
```

1. Download all of our candidate datasets.
2. Instruction Generation (please ref to [CaptionsGenerator](AnyEdit_Collection/diverse_Instruction_generation/README.md)).
3. Pre-filter for target images (before editing)
```bash
CUDA_VISIBLE_DEVICES=2 python pre_filter.py --instruction-path [xx.json] --instruction-type [] --image-root []
```
4. Image Editing (refer to [scripts](scripts) for more examples)
5. Post-filter for final datasets
```bash
CUDA_VISIBLE_DEVICES=2 python post_filter.py --instruction-type []
```
# 🧳 Project Folder Structure

- Datasets/
  - anyedit_datasets/
    - add
    - remove
    - replace
  - coco/
    - train2014/
      - 0.jpg
      - 1.jpg
  - flux_coco_images/
      - 0.jpg
      - 1.jpg
  - add_postfilter.json
  - remove_postfilter.json
  - replace_postfilter.json

# 🎖️ AnyEdit Editing Results (Part Ⅰ)
  |  Original Image   | Edit Type | Edit Instruction  | Edited Image  |
  |  ----  | ----  | ----  | ----  |
  | <img src="figures/example_figures/action_change_origin.jpg" width="250" height="250"> | Action Change | Make the action of the plane to taking off | <img src="figures/example_figures/action_change_edit.jpg" width="250" height="250"> |
  | <img src="figures/example_figures/add_origin.jpg" width="250" height="250"> | Add | Include a candle on top of the cake | <img src="figures/example_figures/add_edit.jpg" width="250" height="250"> |
  | <img src="figures/example_figures/appearance_alter_new_origin.jpg" width="250" height="250"> | Appearance Alter | Make the horses wearing garlands | <img src="figures/example_figures/appearance_alter_new_edit.jpg" width="250" height="250"> |
  | <img src="figures/example_figures/background_change_new_origin.jpg" width="250" height="250"> | Background Change | Alter the background to a garden | <img src="figures/example_figures/background_change_new_edit.jpg" width="250" height="250"> |
  | <img src="figures/example_figures/color_alter_origin.jpg" width="250" height="250"> | Color Alter | Alter the color of frame to orange | <img src="figures/example_figures/color_alter_edit.jpg" width="250" height="250"> |
  | <img src="figures/example_figures/counting_origin.jpg" width="250" height="250"> | Counting | The number of camels increases to two | <img src="figures/example_figures/counting_edit.jpg" width="250" height="250"> |
  | <img src="figures/example_figures/implicit_change_new_origin.jpg" width="250" height="250"> | Implicit Change | What will happen if the sun never go down? | <img src="figures/example_figures/implicit_change_new_edit.jpg" width="250" height="250"> |
  | <img src="figures/example_figures/material_change_origin.jpg" width="250" height="250"> | Material Change | Change the material of kitten like aluminium_foil | <img src="figures/example_figures/material_change_edit.jpg" width="250" height="250"> |
  | <img src="figures/example_figures/movement_origin.jpg" width="250" height="250"> | Movement | Shift the man in the image | <img src="figures/example_figures/movement_edit.jpg" width="250" height="250"> |
  | <img src="figures/example_figures/outpaint_origin.jpg" width="250" height="250"> | Outpaint | Outpaint the image as you can | <img src="figures/example_figures/outpaint_edit.jpg" width="250" height="250"> |
  | <img src="figures/example_figures/relation_origin.jpg" width="250" height="250"> | Relation | Place two yellow flowers in the middle of the table | <img src="figures/example_figures/relation_edit.jpg" width="250" height="250"> |
  | <img src="figures/example_figures/remove_origin.jpg" width="250" height="250"> | Remove | Remove the person on skis | <img src="figures/example_figures/remove_edit.jpg" width="250" height="250"> |
  | <img src="figures/example_figures/replace_origin.jpg" width="250" height="250"> | Replace | Replace the elephant with a seal | <img src="figures/example_figures/replace_edit.jpg" width="250" height="250"> |
  | <img src="figures/example_figures/resize_origin.jpg" width="250" height="250"> | Resize | Zoom out the giraffes in the image | <img src="figures/example_figures/resize_edit.jpg" width="250" height="250"> |
  | <img src="figures/example_figures/rotation_change_origin.jpg" width="250" height="250"> | Rotation Change | Turn the bag counterclockwise | <img src="figures/example_figures/rotation_change_edit.jpg" width="250" height="250"> |
  | <img src="figures/example_figures/style_change_origin.jpg" width="250" height="250"> | Style Change | Change the style of the image to contrast | <img src="figures/example_figures/style_change_edit.jpg" width="250" height="250"> |
  | <img src="figures/example_figures/textual_change_origin.jpg" width="250" height="250"> | Textual Change | Replace the text 'eddie' with 'stobart' | <img src="figures/example_figures/textual_change_edit.jpg" width="250" height="250"> |
  | <img src="figures/example_figures/tune_transfer_origin.jpg" width="250" height="250"> | Tune Transfer | Change the season to autumn | <img src="figures/example_figures/tune_transfer_edit.jpg" width="250" height="250"> |





  
  
  
  # 🎖️ AnyEdit Editing Results (Part Ⅱ)
  |  Original Image   |  Reference Image | Edit Type | Edit Instruction  | Edited Image  |
  |  ----  | ----  | ----  | ----  | ----  |
  | <img src="figures/example_figures/visual_bbox_origin.jpg" width="250" height="250"> | <img src="figures/example_figures/visual_bbox_visual_input.jpg" width="250" height="250"> |Visual Bbox | Follow the given bounding box [v*] to remove the skis | <img src="figures/example_figures/visual_bbox_edit.jpg" width="250" height="250"> |
  | <img src="figures/example_figures/visual_depth_origin.jpg" width="250" height="250"> | <img src="figures/example_figures/visual_depth_visual_input.jpg" width="250" height="250"> |Visual Depth | Refer to the given depth image [v*] to remove umbrella  | <img src="figures/example_figures/visual_depth_edit.jpg" width="250" height="250"> |
  | <img src="figures/example_figures/visual_material_transfer_new_origin.jpg" width="250" height="250"> | <img src="figures/example_figures/visual_material_transfer_new_visual_input.jpg" width="250" height="250"> |Visual Material Transfer | Change the material of monument like linen | <img src="figures/example_figures/visual_material_transfer_new_edit.jpg" width="250" height="250"> |
  | <img src="figures/example_figures/visual_reference_origin.jpg" width="250" height="250"> | <img src="figures/example_figures/visual_reference_visual_input.jpg" width="250" height="250"> |Visual Reference | Replace the elephants to [v*] | <img src="figures/example_figures/visual_reference_edit.jpg" width="250" height="250"> |
  | <img src="figures/example_figures/visual_scribble_origin.jpg" width="250" height="250"> | <img src="figures/example_figures/visual_scribble_visual_input.jpg" width="250" height="250"> |Visual Scribble | Refer to the given scribble [v*] to replace the toilet paper with a book  | <img src="figures/example_figures/visual_scribble_edit.jpg" width="250" height="250"> |
  | <img src="figures/example_figures/visual_segment_origin.jpg" width="250" height="250"> | <img src="figures/example_figures/visual_segment_visual_input.jpg" width="250" height="250"> |Visual Segment | Follow the given segment image [v*] to remove truck  | <img src="figures/example_figures/visual_segment_edit.jpg" width="250" height="250"> |
  | <img src="figures/example_figures/visual_sketch_origin.jpg" width="250" height="250"> | <img src="figures/example_figures/visual_sketch_visual_input.jpg" width="250" height="250"> |Visual Sketch | Watch the given sketch [v*] to replace the bananas to apples | <img src="figures/example_figures/visual_sketch_edit.jpg" width="250" height="250"> |


  