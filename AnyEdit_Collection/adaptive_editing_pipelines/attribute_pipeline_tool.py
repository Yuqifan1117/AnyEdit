import os
from diffusers import DiffusionPipeline
from tools.appearance_alter.pipeline_stable_diffusion_3_instructpix2pix import StableDiffusion3InstructPix2PixPipeline
from diffusers.utils import load_image
from wys_ip2p import LocalEditor
import torch, torchvision
from PIL import Image
import GroundingDINO.groundingdino.datasets.transforms as T
from GroundingDINO.groundingdino.models import build_model
from GroundingDINO.groundingdino.util.slconfig import SLConfig
from GroundingDINO.groundingdino.util.utils import clean_state_dict
from segment_anything import build_sam, SamPredictor
import cv2
import numpy as np
import warnings
import spacy
import json
from tqdm import tqdm
import argparse
import logging
from tool import is_human_variant, return_parameters, maskgeneration
warnings.filterwarnings("ignore")

def load_sam_image(image_path):
    # load image
    image_pil = Image.open(image_path).convert("RGB")  # load image
    image_pil = image_pil.resize((512, 512), resample=Image.Resampling.LANCZOS)
    transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    image, _ = transform(image_pil, None)  # 3, h, w
    return image_pil, image
def load_sam_image_from_Image(image_pil):
    # load image
    transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    image, _ = transform(image_pil, None)  # 3, h, w
    return image

def load_model(model_config_path, model_checkpoint_path, device):
    args = SLConfig.fromfile(model_config_path)
    args.device = device
    model = build_model(args)
    checkpoint = torch.load(model_checkpoint_path, map_location="cpu")
    load_res = model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
    _ = model.eval()
    return model

def generate_tags(raw_text):
    # generate specific categories in the caption by spacy
    nlp = spacy.load("en_core_web_sm")
    tags = {'nouns':[], 'adj':[]}
    words_list = nlp(raw_text)
    for i in range(len(words_list)-1):
        token = words_list[i]
        next_token = words_list[i+1]
        if token.pos_ == 'ADJ' and next_token.pos_ == 'NOUN' and token.text.lower() not in tags['adj']:
            tags['adj'].append(token.text.lower())
        elif token.pos_ == 'NOUN' and token.tag_ != 'VBG' and token.text.lower() not in tags['nouns'] and next_token.text.lower() not in ['of']:
            if next_token.pos_ == 'NOUN':
                tags['nouns'].append(' '.join([token.text.lower(), next_token.text.lower()])) # fine-grained for attribute
                # tags['nouns'].append(next_token.text.lower())
            else:
                tags['nouns'].append(token.text.lower())
    if words_list[-1].pos_ == 'NOUN' and token.text.lower() not in tags['nouns']:
        tags['nouns'].append(words_list[-1].text.lower())
    return tags

def sdxl_pipe(base, refiner, prompt):
    # Define how many steps and what % of steps to be run on each experts (80/20) here
    n_steps = 50
    high_noise_frac = 0.8

    # run both experts
    image = base(
        prompt=prompt,
        num_inference_steps=n_steps,
        denoising_end=high_noise_frac,
        output_type="latent",
    ).images
    image = refiner(
        prompt=prompt,
        num_inference_steps=n_steps,
        denoising_start=high_noise_frac,
        image=image,
    ).images[0]
    return image.resize((512,512), resample=Image.Resampling.LANCZOS)

def cropimage2image(original_image, mask, background_image, scale):
    original_image_array = np.array(original_image)
    mask_array = np.array(mask)
    background_image_array = np.array(background_image)

    coords = np.argwhere(mask_array > 0)
    min_y, min_x = np.min(coords, axis=0)
    max_y, max_x = np.max(coords, axis=0)

    extracted_content = original_image_array[min_y:max_y+1, min_x:max_x+1]
    extracted_content_mask=mask_array[min_y:max_y+1, min_x:max_x+1]

    original_w = max_x - min_x
    original_h = max_y - min_y
    scaled_w = int(original_w * scale)
    scaled_h = int(original_h * scale)
    resized_content = Image.fromarray(extracted_content).resize((scaled_w, scaled_h), resample=Image.Resampling.LANCZOS) 
    resized_content_mask = Image.fromarray(extracted_content_mask).resize((scaled_w, scaled_h), resample=Image.Resampling.LANCZOS)

    result_array = background_image_array
    resized_content_array=np.array(resized_content)
    resized_content_mask_array=np.array(resized_content_mask)
    result_array[min_y:min_y+scaled_h, min_x:min_x+scaled_w][resized_content_mask_array > 0] = resized_content_array[resized_content_mask_array > 0]

    result_image = Image.fromarray(result_array)
    return result_image

def poisson_image_blending(source, mask, destination):
    # navigate the source img location
    source = cv2.cvtColor(np.asarray(source),cv2.COLOR_RGB2BGR)
    destination = cv2.cvtColor(np.asarray(destination),cv2.COLOR_RGB2BGR)
    mask = cv2.cvtColor(np.array(mask),cv2.COLOR_GRAY2BGR)
    width, height, channels = source.shape
    center = (height//2, width//2)

    # using built-in funtion `cv2.seamlessClone` to acommpulish Poisson Image
    blended_image = cv2.seamlessClone(destination, source, mask, center, 2) # cv::MIXED_CLONE = 2
    blended_image = Image.fromarray(cv2.cvtColor(blended_image,cv2.COLOR_BGR2RGB))
    return blended_image

def load_tool_model(args):
    sdxl_ckpt = args.sdxl_ckpt
    ultraedit_ckpt = args.sd3_ckpt
    device = 'cuda'

    det_model = load_model(args.groundingdino_config_file, args.groundingdino_ckpt, device=device)
    sam_model = SamPredictor(build_sam(checkpoint=args.sam_ckpt).to(device))
    sdxl_base = DiffusionPipeline.from_pretrained(
        sdxl_ckpt, torch_dtype=torch.float16, variant="fp16", use_safetensors=True
    ).to(device)
    sdxl_refiner = DiffusionPipeline.from_pretrained(
        sdxl_ckpt,
        text_encoder_2=sdxl_base.text_encoder_2,
        vae=sdxl_base.vae,
        torch_dtype=torch.float16,
        use_safetensors=True,
        variant="fp16",
    ).to(device)

    edit_pipe = StableDiffusion3InstructPix2PixPipeline.from_pretrained(ultraedit_ckpt, torch_dtype=torch.float16).to("cuda:1")
    return sdxl_base, sdxl_refiner, edit_pipe, det_model, sam_model

def clip_filtering_image(image_elements, text, clip_processor, clip_model, device="cuda"):
    clip_inputs = clip_processor(text=[text], images=image_elements, return_tensors="pt").to(device)
    clip_outputs = clip_model(**clip_inputs)
    image_features = clip_outputs['image_embeds'] / clip_outputs['image_embeds'].norm(dim=-1, keepdim=True)
    text_features = clip_outputs['text_embeds'] / clip_outputs['text_embeds'].norm(dim=-1, keepdim=True)
    clip_similarity = 100. * image_features @ text_features.T
    results = torch.max(clip_similarity, dim=0)
    return results[0], results[1].item()

def attribute_pipeline(det_model, sam_model, edit_pipe, input, edited_object, output, init_image_path,
                            edited_image_path_pre, instruction, instruction_type, kernel_size, sdxl_base=None, sdxl_refiner=None):
    if not os.path.exists(init_image_path): 
        init_image = sdxl_pipe(sdxl_base, sdxl_refiner, input)
        init_image.save(init_image_path)
    total_edited_images = []
    original_size = Image.open(init_image_path).convert("RGB").size
    total_objects = generate_tags(input)['nouns']
    if edited_object not in total_objects:
        total_objects.append(edited_object)
    if edited_object.split(' ')[-1] not in total_objects:
        total_objects.append(edited_object.split(' ')[-1])
    prompt = ', '.join(total_objects)
    if instruction_type in ["appearance_alter", "material_alter"]:
        mask_pil, image_pil, _, _ = maskgeneration(det_model, sam_model, init_image_path, prompt,
                                               mask_mode='merge', box_threshold=0.2, text_threshold=0.2, target_object=edited_object)
    elif instruction_type in ["color_alter"]:
        mask_pil, image_pil, _, _ = maskgeneration(det_model, sam_model, init_image_path, prompt,
                                                mask_mode='merge', box_threshold=0.25, text_threshold=0.2, target_object=edited_object)
        if mask_pil is None:
            mask_pil, image_pil, _, _ = maskgeneration(det_model, sam_model, init_image_path, prompt,
                                                mask_mode='merge', box_threshold=0.2, text_threshold=0.2, target_object=edited_object)
    if edited_object.split(' ')[-1] in ['people', 'man', 'woman', 'boy', 'girl', 'person'] or is_human_variant(edited_object.split(' ')[-1]):
        prompt = prompt + ', head, face'
        face_mask_pil, _, _, _ = maskgeneration(det_model, sam_model, init_image_path, prompt,
                                                    mask_mode='merge', box_threshold=0.2, text_threshold=0.2,
                                                    target_object=["face", "head"])
    else:
        face_mask_pil = None
    if mask_pil is None:
        print('Skip: mask is none')
        return None
    else:
        mask_path = edited_image_path_pre.replace("edited_img", "mask") + f'.png'
        mask_pil.save(mask_path)
        cv2_mask_image = cv2.imread(mask_path)
        maskimage_dilate = cv2.dilate(cv2_mask_image, np.ones(kernel_size, np.uint8))
        mask_pil = Image.fromarray(cv2.cvtColor(maskimage_dilate,cv2.COLOR_BGR2GRAY))
        os.remove(mask_path)
    if face_mask_pil is not None:
        axy_path = edited_image_path_pre.replace("edited_img", "mask") + f'_axy.png'
        face_mask_pil.save(axy_path)
        face_cv2_mask_image = cv2.imread(axy_path)
        face_maskimage_dilate = cv2.dilate(face_cv2_mask_image, np.ones(kernel_size, np.uint8))
        face_maskimage_dilate = Image.fromarray(cv2.cvtColor(face_maskimage_dilate,cv2.COLOR_BGR2GRAY))
        noface_mask_pil = Image.fromarray(np.where(np.asarray(face_maskimage_dilate)==0, True, False))
        mask_pil = Image.fromarray(np.where(np.asarray(mask_pil), True, False) & np.asarray(noface_mask_pil))
        os.remove(axy_path)
    for i in range(1):
        image = edit_pipe(
            instruction,
            image=image_pil.convert("RGB"),
            mask_img=mask_pil.convert("RGB"),
            negative_prompt="bad quality, text",
            num_inference_steps=50,
            image_guidance_scale=1.5,
            guidance_scale=8.0,
        ).images[0]
            
        
        edited_image = image.resize(original_size, resample=Image.Resampling.LANCZOS)
        save_edited_image_path = edited_image_path_pre + f'_{i}.png'
        edited_image.save(save_edited_image_path)
        
        mask_path = edited_image_path_pre.replace("edited_img", "mask") + f'.png'
        mask_pil.resize(original_size, resample=Image.Resampling.LANCZOS).save(mask_path)

        
        cur_data = {"edit": instruction, "edit object": edited_object, "output": output, "input": input, "edit_type": instruction_type, "image_file": init_image_path, "edited_file": save_edited_image_path}
        total_edited_images.append(cur_data)
    return total_edited_images 


def parse_args():
    parser = argparse.ArgumentParser(description="Editing Pipeline")
    parser.add_argument("--groundingdino-config-file", required=True, help="path to groundingdino config file.")
    parser.add_argument("--groundingdino-ckpt", required=True, help="path to groundingdino model file")
    parser.add_argument("--sam-ckpt", required=True, help="path to sam model file")
    parser.add_argument("--sdxl-ckpt", required=True, help="path to sdxl model file")
    parser.add_argument("--sd3-ckpt", required=True, help="path to finetuned sd3 model file for attribute")
    parser.add_argument("--instruction-path", required=True, help="path to instruction path")
    parser.add_argument("--instruction-type", required=True, help="specify the instruction type.")
    parser.add_argument("--json-path", default=None)
    parser.add_argument("--image-path", default=None)
    parser.add_argument("--idx", type=int, default=0, help="specify the experiment id.")
    parser.add_argument(
        "--options",
        nargs="+",
        help="in xxx=yyy format will be merged into config file (deprecate), "
        "change to --cfg-options instead.",
    )
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()

    try:
        with open(args.json_path) as f:
            edit_instruction_data = json.load(f)
    except:
        with open(args.json_path, 'r') as f:
            edit_instruction_data = [json.loads(line) for line in f]

    os.makedirs(f'{args.instruction_path}/{args.instruction_type}/mask', exist_ok=True)
    os.makedirs(f'{args.instruction_path}/{args.instruction_type}/heatmap', exist_ok=True)
    os.makedirs(f'{args.instruction_path}/{args.instruction_type}/edited_img', exist_ok=True)
    os.makedirs(f'{args.instruction_path}/{args.instruction_type}/input_img', exist_ok=True)

    sdxl_base, sdxl_refiner, edit_pipe, det_model, sam_model = load_tool_model(args)
    valid_number = 0
    success_data = []
    failure_data = []
    final_edited_results = []
    for data in tqdm(edit_instruction_data):
        instruction_type = data['edit_type']
        if 'input' not in data or 'output' not in data:
            print(f'Skip: input or output in data')
            continue

        if instruction_type in ["material_alter", "appearance_alter", "color_alter"]:
            if 'edited object' not in data:
                print(f'Skip: no edited object in data')
                continue
            input, edited_object, output, init_image_path, instruction, edited_image_path_pre = \
                return_parameters(data, args, init_image_root=args.image_path)
            if edited_object not in input:
                print(f'Skip: no edited object in {init_image_path}')
                failure_data.append(data)
                continue

            have_exist = False
            if os.path.exists(edited_image_path_pre + f'.png'):
                save_edited_image_path = edited_image_path_pre + f'.png'
                have_exist = True
            else:
                for i in range(3):
                    if os.path.exists(edited_image_path_pre+f'_{i}.png'):
                        save_edited_image_path = edited_image_path_pre+f'_{i}.png'
                        have_exist = True
                        break
            if have_exist:
                valid_number += 1
                edit_image = {"edit": instruction, "edit object": edited_object, "output": output, "input": input, "edit_type": instruction_type, "image_file": init_image_path, "edited_file": save_edited_image_path}
                final_edited_results.append(edit_image)
                success_data.append(data)
                print(f'Skip: image existed')
                continue

            if instruction_type in ["appearance_alter"]:
                kernel_size = (30, 30)
            elif instruction_type in ["color_alter", "material_alter"]:
                kernel_size = (15, 15)
            else:
                raise NotImplementedError
            edit_image = attribute_pipeline(det_model=det_model, sam_model=sam_model,edit_pipe=edit_pipe,
                                            input=input, edited_object=edited_object, output=output,
                                            init_image_path=init_image_path, edited_image_path_pre=edited_image_path_pre,
                                            instruction=instruction, instruction_type=instruction_type, kernel_size=kernel_size, sdxl_base=sdxl_base, sdxl_refiner=sdxl_refiner)
        else:
            raise NotImplementedError

        if edit_image is not None:
            valid_number += 1
            final_edited_results.extend(edit_image)
            success_data.append(data)
        else:
            failure_data.append(data)
    # logging.info(f"valid editing insturction data: {valid_number}")
    print(f"valid editing insturction data: {valid_number}")
    with open(f'{args.instruction_path}/{args.instruction_type}/final_edit_results_{args.idx}.json', 'w') as results_file:
        json.dump(final_edited_results, results_file, indent=4)
    with open(f'{args.instruction_path}/{args.instruction_type}/edit_success_{args.idx}.json', 'w') as success_file:
        json.dump(success_data, success_file, indent=4)
    with open(f'{args.instruction_path}/{args.instruction_type}/edit_failure_{args.idx}.json', 'w') as failure_file:
        json.dump(failure_data, failure_file, indent=4)