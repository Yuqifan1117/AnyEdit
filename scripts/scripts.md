# Examples

```bash
bash [pipeline].sh [gpu_id]
```

## Text Input

#### local pipeline (1)

```bash
bash scripts/add.sh 2
bash scripts/remove.sh 2
bash scripts/counting.sh 4  # number是原有的数量
bash scripts/background_change.sh 2
bash scripts/replace.sh 2

# inpaint pipeline (1), 存在一点问题，因为现在的方式不是图层的形式，因此移动或变化大小后，背景会出现空白等不一致的问题
bash scripts/resize.sh 2
bash scripts/movement.sh 2
```

#### gloabl pipeline (2,3,4) 

```bash
# pipeline 2
bash scripts/appearance_alter.sh 2
bash scripts/material_alter.sh 2 # text-driven, material_transfer is image driven
# pipeline 3
bash scripts/color_alter.sh 2
bash scripts/tune_transfer.sh 2
```

#### fine-grained pipeline (5)

```bash
bash scripts/action_change.sh 2  
# the case of emotional_expression is also in action_change
```

#### Prisma (8)

style transfer, please refer to this [repo]()

#### MVImageNet (9)
1. Download [MVImageNet](https://github.com/GAP-LAB-CUHK-SZ/MVImgNet) following their link, Please fill out this [form](https://docs.google.com/forms/d/e/1FAIpQLSfU9BkV1hY3r75n5rc37IvlzaK2VFYbdsvohqPGAjb2YWIbUg/viewform?usp=sf_link) to get the download link and password, then unzip 
2. Folder structure
```shell
|-- ROOT
    |-- class_label
        |-- instance_id
            |-- images
            |-- sparse/0
                |-- cameras.bin   # COLMAP reconstructed cameras
                |-- images.bin    # binary data of input images
                |-- points3D.bin  # COLMAP reconstructed sparse point cloud (not dense) 
```
3. Update the parameters
```bash
cd ROOT/class_label/instance_id
mkdir text_annotations
```
- Replace '--input_model' to the path of 'sparse/0', '--output_model' to the path of 'text_annotations'
- Replace '--text_file' to the path of 'xxx/text_annotations/image.txt', '--image_root' to the path of 'xxx/images'
- Update the category of this image (MVImageNet contain 180 categories)
```bash
bash scripts/rotation_change.sh 2
```

#### groundDINO+crop(10)

```bash
bash scripts/outpainting.sh 2
```

#### Implicit (14)

namely Commonsense Reasoning

move `text_gen_full.json` geneated by [xxxx]() to `./edit_generated_datasets/example_dataset/implicit`

```bash
PYTHONPATH='./' CUDA_VISIBLE_DEVICES=2,3 python implicit_t2i.py  # need 2×24G GPU
```

#### composition&relation (15)

only support right and left for relation

```bash
CUDA_VISIBLE_DEVICES=2 python composition_image_generation.py
```

#### textual (16)

use FLUX

```shell
# input and output should be generated in different time
PYTHONPATH='./' python -W ignore textual.py --gpu-id 7 --mode 'output'
```

## Visual+Text Input

#### other visual input 

(scribble, segmentation, depth, canny); and (openpose, action) is poor (11)

```bash
PYTHONPATH='./' python other_visual_reference_tool.py
```

#### visual_material_transfer (12)

```bash
bash scripts/visual_material_transfer.sh 2
```

#### visual_reference (13)

```bash
bash scripts/visual_image_reference.sh 2
```

