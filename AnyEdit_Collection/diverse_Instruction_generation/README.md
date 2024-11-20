# Diverse Instruction Generation

## File Tree

```smail
└── InstructionGen
    ├── concept
    │   ├── edit.py
    │   └── xxx.py
    ├── checkpoints
    │   ├── llama3-8b
    │   └── mistral-7b
    └── captions_generator.py
```

## Pipeline

| ID   | **Type**                 | **Details**                                                  |
| ---- | ------------------------ | ------------------------------------------------------------ |
| 1    | 1 concept + 1 background | 1. Generate about 5 background for  each concept 2. give instruction for each 1 concept + 1 background |
| 2    | 2 concept composition    |                                                              |
| 3    | Anyedit Instruction      |                                                              |
| 4    | Implicit                 |                                                              |

## Concept

### Environment

```shell
bash setup.sh
```

### Download the concept pool

| version                                                      | description                                                  |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| [classname.json ](https://drive.google.com/drive/folders/17fSf4Dr-_lfTHIt5_xnPzBsg3nr8XKOM) | 207个[图片分类数据](https://paperswithcode.com/datasets?task=image-classification)收集得到的 23,028 个概念， 主要来自[CIFAR10](https://paperswithcode.com/dataset/cifar-10),  [CIFAR100](https://paperswithcode.com/dataset/cifar-100), [ImageNet1k](),[CUB-200](https://paperswithcode.com/dataset/cub-200-2011),[Oxford 102 Flower](https://paperswithcode.com/dataset/oxford-102-flower), [STL10](https://paperswithcode.com/dataset/stl-10), [food101](https://paperswithcode.com/dataset/food-101), [sun397](https://paperswithcode.com/dataset/sun397), [Sports10](https://paperswithcode.com/dataset/sports10),[CNFOOD-241](https://data.mendeley.com/datasets/fspyss5zbb/1), [ifood251](https://github.com/karansikka1/iFood_2019), [Grocery Store](https://github.com/marcusklasson/GroceryStoreDataset) , [voc2007](https://paperswithcode.com/dataset/pascal-voc-2007), [Oxford Pets](https://www.robots.ox.ac.uk/~vgg/data/pets/), [resisc45](https://paperswithcode.com/dataset/resisc45), [kinetics700](https://paperswithcode.com/dataset/kinetics-700), [ucf101](https://www.crcv.ucf.edu/data/UCF101.php)，[openimageV7](https://storage.googleapis.com/openimages/web/factsfigures_v7.html#class-definitions), [TecentMI](https://github.com/Tencent/tencent-ml-images/blob/master/data/dictionary_and_semantic_hierarchy.txt), [iCartoonFace](https://github.com/luxiangju-PersonAI/iCartoonFace?tab=readme-ov-file#Dataset) |
| [SynCLR_concept+background.json](https://drive.google.com/drive/folders/17fSf4Dr-_lfTHIt5_xnPzBsg3nr8XKOM)   [SynCLR_concept.json](https://drive.google.com/drive/folders/17fSf4Dr-_lfTHIt5_xnPzBsg3nr8XKOM) | 使用[SynCLR](https://github.com/google-research/syn-rep-learn/tree/main/SynCLR/synthesis/syn_text)中的`imgnet21k_combined_background_dict`和`imgnet_background_dict`得到 $13,890$ 个概念, 每个concept配了约100个background |
| [concept.json](https://drive.google.com/drive/folders/17fSf4Dr-_lfTHIt5_xnPzBsg3nr8XKOM) | 上面两个文件合成的$35,299$个概念                             |
| [concept_pool(2409 version).json](https://drive.google.com/drive/folders/17fSf4Dr-_lfTHIt5_xnPzBsg3nr8XKOM?ths=true) | 人工筛完的concept                                            |
| 目前的                                                       | GPT再筛选掉                                                  |
|                                                              |                                                              |

## example

```shell
python example.py
```

## 1,2 Caption Generation

背景增广，并且保证多样性。 平均背景数量为 50，太多了可能会影响质量

冗余的caption无法生成，对于编辑来说可能是种毒害，因此caption的生成尽量简单

```shell
python captions_generator.py --gpu-id 4 # 3种mode 'c2b', 'c2cap', 'cb2cap'
```

## 3 Edit Instruction

| dataset name                                                 | support type                        |
| ------------------------------------------------------------ | ----------------------------------- |
| [COCO](), [LAION]()                                          | add, remove, replace, action_change |
| [AnyWord](https://modelscope.cn/datasets/iic/AnyWord-3M/files)(Art, ): ocr_data/Art/data.json重命名为textual_art.json其他类似 | textual (only supporting English)   |
|                                                              |                                     |

```shell
# single turn + batch
PYTHONPATH='./' python edit_instruction/instruction_gen.py --instruction-type textual --gpu-id 6 --batch-size 6 --source-data icdar2017rctw
# multi_turn
PYTHONPATH='./' python edit_instruction/instruction_gen_multi_turn.py --instruction-type mix --gpu-id 4 --iter 3
```

## 4 Implict

```bash
# generate instruction
PYTHONPATH='./' python implicit/instruction_gen.py
# transform the text prompt provided by GPT into dict
PYTHONPATH='./' python implicit/deal_text2json.py
# text_gen_full.json to Anyeidt and run t2i

huggingface-cli download Efficient-Large-Model/Llama-3-VILA1.5-8B --local-dir . --local-dir-use-symlinks False
```

# 参考

其他造caption的文章：

[1] Synth2: Boosting Visual-Language Models with Synthetic Captions and Image Embeddings object

[2] MetaCLIP

[3] SynCLR

[4] [SynthCLIP](https://github.com/hammoudhasan/SynthCLIP/tree/main/TextGen)
