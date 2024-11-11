# AnyEdit
Robust SOTA image editing pipelines for various change types like human imagination.

# Instruction Format
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
