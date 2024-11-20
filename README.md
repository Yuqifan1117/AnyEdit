# AnyEdit
Mastering Unified High-Quality Image Editing for Any Idea
# 🌍 Introduction
AnyEdit is a comprehensive multimodal instruction editing dataset, comprising 2.5 million high-quality editing pairs spanning over 20 editing types across five domains. We ensure the diversity and quality of the AnyEdit collection through three aspects: initial data diversity, adaptive editing process, and automated selection of editing results. Using the dataset, we further train a novel AnyEdit Stable Diffusion with task-aware routing and learnable task embedding for unified image editing. Comprehensive experiments on three benchmark datasets show that AnyEdit consistently boosts the performance of diffusion-based editing models. This presents prospects for developing instruction-driven image editing models that support human creativity.

# 💡 Overview
![image](figures/main_figures/final_example_00.png)
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
  ![image](figures/pipelines.png)

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
  |  Original Image   | Original Caption  | Edit Type | Edit Instruction  | Edited Image  |
  |  ----  | ----  | ----  | ----  | ----  |
  | ![image](figures/add_ori.jpg) | a garage with signs hanging in front of it | add | Add a vintage bicycle parked in front of the garage. | ![image](figures/add_tar.jpg) |
  |![image](figures/remove_origin.jpg) | xxxxx | remove | Delete the train. | ![image](https://github.com/Yuqifan1117/EditAnyPipeline/assets/48062034/e2fcbbbb-c700-4282-9396-c2db64c2f70b) |
  |![image](https://github.com/Yuqifan1117/EditAnyPipeline/assets/48062034/e23070a4-c30d-4d0b-aab2-d21aa97e0f08) | xxxxx | replace | Alter the bicycle to a goat. | ![image](https://github.com/Yuqifan1117/EditAnyPipeline/assets/48062034/da34c506-d354-49a5-ab7a-8dae0a6a7880)|
  |![image](https://github.com/Yuqifan1117/EditAnyPipeline/assets/48062034/321f6e24-4015-49a7-b1d6-a03ce8c861be) | xxxxx | action_change | Change the action of cat to sit | ![image](https://github.com/Yuqifan1117/EditAnyPipeline/assets/48062034/ebb043a0-7127-4d2b-82a7-997695c39627) |
  |![016](https://github.com/Yuqifan1117/EditAnyPipeline/assets/48062034/8c2612e4-8fe3-4d2e-b6bd-92de517f2c77) | xxxxx | rotation_change | Turn the skateboard to the right (clockwise) | ![029](https://github.com/Yuqifan1117/EditAnyPipeline/assets/48062034/2d641c08-ffbc-40cf-b649-dde170d96c04) |
  | ![image](https://github.com/DCDmllm/AnyEdit/assets/48062034/51eb9869-1a71-4f24-b9e9-cc5d1c41374f) | xxxxx | background_change | Turn the background to the city | ![image](https://github.com/DCDmllm/AnyEdit/assets/48062034/d017f46a-cf46-4e6a-b187-4e775521d553) |
  | ![image](https://github.com/DCDmllm/AnyEdit/assets/48062034/aca2fddc-9e0b-4467-a3c0-e53ffdbd0663) | xxxxx | color_alter | Change the color of kites to white | ![image](https://github.com/DCDmllm/AnyEdit/assets/48062034/cf67025e-79c2-4040-8158-217cbb7785ad) |
  | ![image](https://github.com/DCDmllm/AnyEdit/assets/48062034/598443a6-26f8-4432-9453-af79b1aeea03) | xxxxx | appearance_alter | make the leave glow to flower | ![image](https://github.com/DCDmllm/AnyEdit/assets/48062034/2e0639cb-e81b-4b9e-afe3-2000cfa15592) |
  | ![image](https://github.com/DCDmllm/AnyEdit/assets/48062034/772349ca-2b22-499e-b661-1e899fdb7e62) | xxxxx | tune_transfer | Change the time to dusk | ![image](https://github.com/DCDmllm/AnyEdit/assets/48062034/044c9b03-a2cb-4526-9a25-d8fa884d4574)|
  |![image](https://github.com/Yuqifan1117/AnyEdit/assets/48062034/5faa4044-d9bc-4ce0-a550-318f08979cfb) | A woman in a floral swimsuit holds a pink umbrella. | outpainting | / | ![COCO_train2014_000000000036](https://github.com/Yuqifan1117/AnyEdit/assets/48062034/2867098e-41f9-4f05-8638-967ef055b3e8) |
  |![](./edit_generated_datasets/example_dataset/implicit/generated_img/group_0/sample7_ori.png) | / | implicit | What will happen if someone steps on it | ![](./edit_generated_datasets/example_dataset/implicit/generated_img/group_0/sample7_tar.png) |
  |![](./edit_generated_datasets/example_dataset/composition/input.png) |  | composition&relation | change dog 'in the left’ to 'in the right' | ![](./edit_generated_datasets/example_dataset/composition/output.png) |
  |![](./edit_generated_datasets/example_dataset/textual/output/input_2023_gt_4995.jpg) | | Textual | Replace the text 'WILTONS' with 'WILLIAM'S' | ![](./edit_generated_datasets/example_dataset/textual/output/output_2023_gt_4995.jpg) |
  
  
  
  # 🎖️ AnyEdit Editing Results (Part Ⅱ)
  |  Original Image   | Original Caption  | Reference Image | Edit Type | Edit Instruction  | Edited Image  |
  |  ----  | ----  | ----  | ----  | ----  | ----  |
  |![image](https://github.com/Yuqifan1117/EditAnyPipeline/assets/48062034/24be1df7-2048-4858-9ff4-1a4bb4a82448) | a dog on the grass | ![image](https://github.com/Yuqifan1117/EditAnyPipeline/assets/48062034/998d46c9-ac4d-44ab-8594-bc6ba8833158) | visual_reference_replace | replace the dog with [V*] | ![image](https://github.com/Yuqifan1117/EditAnyPipeline/assets/48062034/288987f8-d2ed-4242-85b8-6fd2a2218052) |
  |![image](https://github.com/Yuqifan1117/EditAnyPipeline/assets/48062034/a2bc5871-5076-4801-b664-4768bdafa680) | xxxx | ![image](https://github.com/Yuqifan1117/EditAnyPipeline/assets/48062034/f7101b7b-b58d-42e0-bcab-a2a8d366edee)| image_reference? | replace the dog with [V*] | ![image](https://github.com/Yuqifan1117/EditAnyPipeline/assets/48062034/cd90536c-a4a2-4d1a-a98a-300fd9d15215)|
  |![image](https://github.com/Yuqifan1117/EditAnyPipeline/assets/48062034/a2bc5871-5076-4801-b664-4768bdafa680) | xxxx | ![image](https://github.com/Yuqifan1117/AnyEdit/assets/48062034/dccc130d-91aa-4862-b765-4b5676b10dac) | other::canny_reference | replace the dog with the object corresponding to the [V*] sketch | ![image](https://github.com/Yuqifan1117/EditAnyPipeline/assets/48062034/cd90536c-a4a2-4d1a-a98a-300fd9d15215)|
  | ![image](https://github.com/Yuqifan1117/AnyEdit/assets/48062034/13cd820f-4963-4987-9a21-238f63e10290) | a cat wearing a hat sits in the bar. | ![image](https://github.com/Yuqifan1117/AnyEdit/assets/48062034/2e582356-432b-4871-b106-84b37b7f067d) | mask_reference? | change the color of the mask region to blue | ![image](https://github.com/Yuqifan1117/AnyEdit/assets/48062034/30d1f80e-88d5-420e-9d5e-36a61c954f73) |
  |![000000000394](https://github.com/Yuqifan1117/AnyEdit/assets/48062034/8c045be0-943f-4720-94b8-c2e0aea40134) | A dog was holding a purple frisbee in the grass | ![](./edit_generated_datasets/example_dataset/other_visual/image.png) | other::depth_reference | Follow the depth map [V*] to remove the object | ![remove_000000000394](https://github.com/Yuqifan1117/AnyEdit/assets/48062034/413a9970-eb06-46a1-8f21-2f98be8d8822) |
  |![image](https://github.com/Yuqifan1117/EditAnyPipeline/assets/48062034/d08b2e33-c748-4e4c-9163-50bf900f763a)| xxxx | ![image](https://github.com/Yuqifan1117/EditAnyPipeline/assets/48062034/4ab68ea8-79f6-41f9-9749-623dda1e86dd)| visual_add_replace? | add the [V*] in the image | ![image](https://github.com/Yuqifan1117/EditAnyPipeline/assets/48062034/5ae7e4e2-4990-4a75-bf60-11388501d4c5)|
  |![image](https://github.com/DCDmllm/AnyEdit/assets/48062034/8d20e851-51db-47e5-a72a-682916e1baa1) | A pumpkin is on the table. | ![image](https://github.com/DCDmllm/AnyEdit/assets/48062034/9485895c-992d-453f-9c05-5ceaee8774f7) | visual_material_transfer | transfer the material of [V*] to the pumpkin. | ![image](https://github.com/DCDmllm/AnyEdit/assets/48062034/f9c84eb4-acdd-47bf-aec3-9c01c14b98e8) |
  |![000000134754](https://github.com/Yuqifan1117/AnyEdit/assets/48062034/e12c6d47-87fa-4feb-9271-7f4606d38a5f) | a man and a woman walk on the beach with surfboards | ![color_alter_000000134754](https://github.com/Yuqifan1117/AnyEdit/assets/48062034/a67834a7-d012-4e99-b465-57f933c96db5) | heatmap_reference? | Follow the heatmap [V*] to change the color |![color_alter_000000134754_focus_0](https://github.com/Yuqifan1117/AnyEdit/assets/48062034/4fc09427-f633-4f4d-8e23-096a7b1f691f)|
  | |  | ![](./edit_generated_datasets/example_dataset/other_visual/test_seg.png) | other::segementation |  |![](./edit_generated_datasets/example_dataset/other_visual/input_img/house.png)|
  | |  | ![](./edit_generated_datasets/example_dataset/other_visual/test_scr.png) | other::scribble |  |![](./edit_generated_datasets/example_dataset/other_visual/input_img/bag.png)|
  | ![](./edit_generated_datasets/example_dataset/tryon/02532_00.jpg) | | ![](./edit_generated_datasets/example_dataset/tryon/14096_00.jpg)![](./edit_generated_datasets/example_dataset/tryon/14096_00_cloth.jpg) | tryon | |![](./edit_generated_datasets/example_dataset/tryon/02532_00_14096_00.jpg)|
  