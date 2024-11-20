import os
from diffusers import DiffusionPipeline, StableDiffusionInpaintPipeline, StableDiffusionPipeline, AutoPipelineForText2Image, FluxPipeline
import torch
from PIL import Image
import random
import GroundingDINO.groundingdino.datasets.transforms as T
from GroundingDINO.groundingdino.models import build_model
from GroundingDINO.groundingdino.util.slconfig import SLConfig
from GroundingDINO.groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap
from segment_anything import build_sam, SamPredictor
import cv2
import numpy as np
import warnings
import json
from tqdm import tqdm
import argparse
import spacy
from tool import is_human_variant, return_parameters, maskgeneration
from termcolor import cprint
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
    checkpoint = torch.load(model_checkpoint_path, map_location="cuda")
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

def sdxl_pipe(base, prompt):
    # Define how many steps and what % of steps to be run on each experts (80/20) here
    # n_steps = 50
    # high_noise_frac = 0.8
    # prompt = prompt.strip('.')
    # prompt = f'{prompt}, high resolution, realistic'
    # # run both experts
    # image = base(
    #     prompt=prompt,
    #     num_inference_steps=n_steps,
    #     denoising_end=high_noise_frac,
    #     output_type="latent",
    # ).images
    # image = refiner(
    #     prompt=prompt,
    #     num_inference_steps=n_steps,
    #     denoising_start=high_noise_frac,
    #     image=image,
    # ).images[0]

    # sdxl-turbo, but not good for human
    # image = base(prompt=prompt, num_inference_steps=4, guidance_scale=0.0).images[0]
    
    # flux
    seed = random.randint(0, 2<<32)
    image = base(
        prompt,
        output_type="pil",
        num_inference_steps=4, #use a larger number if you are using [dev]
        generator=torch.Generator("cuda").manual_seed(seed)
    ).images[0]
    return image.resize((512,512), resample=Image.Resampling.LANCZOS)

def sd_pipe_generation(sd_model, prompt):
    image = sd_model(prompt).images[0]  
        
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
    resized_content = Image.fromarray(extracted_content).resize((scaled_w, scaled_h), resample=Image.Resampling.LANCZOS) #(max_x - min_x + 1, max_y - min_y + 1))
    resized_content_mask = Image.fromarray(extracted_content_mask).resize((scaled_w, scaled_h), resample=Image.Resampling.LANCZOS) #(max_x - min_x + 1, max_y - min_y + 1))

    result_array = background_image_array
    resized_content_array=np.array(resized_content)
    resized_content_mask_array=np.array(resized_content_mask)
    result_array[min_y:min_y+scaled_h, min_x:min_x+scaled_w][resized_content_mask_array > 0] = resized_content_array[resized_content_mask_array > 0]

    result_image = Image.fromarray(result_array)
    return result_image

def resize_cropimage2image(original_image, mask, background_image, x, y, scale):
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
    resized_content = Image.fromarray(extracted_content).resize((scaled_w, scaled_h), resample=Image.Resampling.LANCZOS) #(max_x - min_x + 1, max_y - min_y + 1))
    resized_content_mask = Image.fromarray(extracted_content_mask).resize((scaled_w, scaled_h), resample=Image.Resampling.LANCZOS) #(max_x - min_x + 1, max_y - min_y + 1))

    result_array = background_image_array
    resized_content_array=np.array(resized_content)
    resized_content_mask_array=np.array(resized_content_mask)
    result_array[y:y+scaled_h, x:x+scaled_w][resized_content_mask_array > 0] = resized_content_array[resized_content_mask_array > 0]

    result_image = Image.fromarray(result_array)
    return result_image

def load_tool_model(args):
    config_file = args.groundingdino_config_file #'GroundingDINO/groundingdino/config/GroundingDINO_SwinB_cfg.py' 
    grounded_checkpoint = args.groundingdino_ckpt #'checkpoints/groundingdino_swinb_cogcoor.pth'  
    sam_checkpoint = args.sam_ckpt #'checkpoints/sam_vit_h_4b8939.pth'
    sdxl_ckpt = args.sdxl_ckpt # "stabilityai/sdxl-turbo"
    inpaint_ckpt = args.inpaint_ckpt 
    device = 'cuda'

    det_model = load_model(config_file, grounded_checkpoint, device=device)
    sam_model = SamPredictor(build_sam(checkpoint=sam_checkpoint).to(device))


    if 'FLUX' in sdxl_ckpt or 'Flux' in sdxl_ckpt or 'flux' in sdxl_ckpt: 
        # for convience, we use sdxl_base varient
        sdxl_base = FluxPipeline.from_pretrained(sdxl_ckpt, torch_dtype=torch.bfloat16).to(device)
    else:
        sdxl_base = AutoPipelineForText2Image.from_pretrained(sdxl_ckpt, torch_dtype=torch.float16, variant="fp16").to(device)

    inpaint_pipe = StableDiffusionInpaintPipeline.from_pretrained(
        inpaint_ckpt,
        variant="fp16",
        torch_dtype=torch.float16,
    ).to(device)

    # sdxl_base = None
    # sdxl_refiner = None
    return sdxl_base, inpaint_pipe, det_model, sam_model

def clip_filtering_image(image_elements, text, clip_processor, clip_model, device="cuda"):
    clip_inputs = clip_processor(text=[text], images=image_elements, return_tensors="pt", padding=True).to(device)
    clip_outputs = clip_model(**clip_inputs)
    image_features = clip_outputs['image_embeds'] / clip_outputs['image_embeds'].norm(dim=-1, keepdim=True)
    text_features = clip_outputs['text_embeds'] / clip_outputs['text_embeds'].norm(dim=-1, keepdim=True)
    clip_similarity = 100. * image_features @ text_features.T
    results = torch.max(clip_similarity, dim=0)
    return results[0], results[1].item()


def add(det_model, sam_model, inpaint_pipe, edited_object, output, instruction,
        edited_image_path_pre, sdxl_base=None):
    # generate target image: y
    mask_pil = None
    total_objects = generate_tags(output)['nouns']
    if edited_object not in total_objects:
        total_objects.append(edited_object)
        total_objects.append(edited_object.split(' ')[-1])
    if mask_pil is None:
        # target_image = sd_pipe_generation(sd_pipe, output)
        target_prompt = f"{output}"
        target_image = sdxl_pipe(sdxl_base, target_prompt)
        prompt = ', '.join(total_objects)
        mask_pil, image_pil, mask_bbox, union_region = maskgeneration(det_model, sam_model, target_image, prompt,
                                                                      mask_mode='max', box_threshold=0.2,text_threshold=0.2, target_object=edited_object)
        if mask_pil is None:
            edit_scene = ' '.join(instruction.split(' ')[1:])
            target_prompt = f"{edited_object}, {edit_scene}"
            target_image = sdxl_pipe(sdxl_base, target_prompt)
            mask_pil, image_pil, mask_bbox, union_region = maskgeneration(det_model, sam_model, target_image, prompt,
                                                                        mask_mode='max', box_threshold=0.2,text_threshold=0.2, target_object=edited_object)
    if mask_pil is None:
        print('Skip: Generation mask fails')
        return None

    mask_path = edited_image_path_pre.replace("edited_img", "mask") + f'.png'
    mask_pil.save(mask_path)
    mask_bbox_path = edited_image_path_pre.replace("edited_img", "mask") + f'_bbox.png'
    mask_bbox.save(mask_bbox_path)

    cv2_mask_image = cv2.imread(mask_path)
    # more mask region for remove artifact
    if union_region < 0.33:
        maskimage_dilate = cv2.dilate(cv2_mask_image, np.ones((30, 30), np.uint8))
    elif union_region < 0.66:
        maskimage_dilate = cv2.dilate(cv2_mask_image, np.ones((30, 30), np.uint8))
    else:
        maskimage_dilate = cv2.dilate(cv2_mask_image, np.ones((30, 30), np.uint8))  
    maskimage_dilate = Image.fromarray(cv2.cvtColor(maskimage_dilate,cv2.COLOR_BGR2GRAY))
    negative_prompt = f'{edited_object}, other objects, person, text, watermark, username'
    total_edited_images = []
    for i in range(1):
        init_image_input = inpaint_pipe(prompt="", negative_prompt=negative_prompt, image=image_pil, mask_image=maskimage_dilate).images[0]
        init_mask_pil, _, _, _ = maskgeneration(det_model, sam_model, init_image_input, prompt, mask_mode='max', target_object=edited_object)
        if init_mask_pil is None:  # remove the edited object
            cur_init_image_path = edited_image_path_pre.replace("edited_img", "input_img") + f'.png'
            init_image_input.save(cur_init_image_path)
            save_edited_image_path = edited_image_path_pre + f'.png'

            target_image.save(save_edited_image_path)
            cur_data = {"edit": instruction, "edit object": edited_object, "output": output,
                        "input": None, "edit_type": "add", "image_file": cur_init_image_path, "edited_file": save_edited_image_path}
            total_edited_images.append(cur_data)
        else:
            intersection_mask = np.asarray(mask_pil) & np.asarray(init_mask_pil)
            num_inter_ones = np.sum(intersection_mask == 1)
            num_original_ones = np.sum(np.asarray(mask_pil)== 1)
            percentage = (num_inter_ones / num_original_ones)
            if percentage < 0.5:
                cur_init_image_path = edited_image_path_pre.replace("edited_img", "input_img") + f'.png'
                init_image_input.save(cur_init_image_path)
                save_edited_image_path = edited_image_path_pre + f'.png'

                target_image.save(save_edited_image_path)
                cur_data = {"edit": instruction, "edit object": edited_object, "output": output,
                            "input": None, "edit_type": "add", "image_file": cur_init_image_path, "edited_file": save_edited_image_path}
                total_edited_images.append(cur_data)                
        # bbox inpaint
        # init_image_input_bbox = inpaint_pipe(prompt="", negative_prompt=negative_prompt, image=image_pil, mask_image=mask_bbox).images[0]
        # init_mask_pil_bbox, _, _, _ = maskgeneration(det_model, sam_model, init_image_input_bbox, prompt, mask_mode='max', target_object=edited_object)
        # if init_mask_pil_bbox is None:  # remove the edited object
        #     cur_init_image_path = edited_image_path_pre.replace("edited_img", "input_img") + f'_bbox.png'
        #     init_image_input.save(cur_init_image_path)
        #     save_edited_image_path = edited_image_path_pre + f'_bbox.png'

        #     target_image.save(save_edited_image_path)
        #     cur_data = {"edit": instruction, "edit object": edited_object, "output": output,
        #                 "input": None, "edit_type": "add", "image_file": cur_init_image_path, "edited_file": save_edited_image_path}
        #     total_edited_images.append(cur_data)
    return total_edited_images 

def remove(det_model, sam_model, inpaint_pipe, input, output, edited_object, init_image_path, instruction,
           edited_image_path_pre, sdxl_base=None, mask_mode='max', remove_number=1):
    # generate init image: x
    if not os.path.exists(init_image_path): 
        init_image = sdxl_pipe(sdxl_base, input)
        init_image.save(init_image_path)

    original_size = Image.open(init_image_path).convert("RGB").size
    total_objects = generate_tags(input)['nouns']
    if edited_object not in total_objects:
        total_objects.append(edited_object)
    prompt = ', '.join(total_objects)
    mask_pil, image_pil, mask_bbox, union_region = maskgeneration(det_model, sam_model, init_image_path, prompt,
                                                                  mask_mode=mask_mode, target_object=edited_object)

    if mask_pil is None or union_region > 0.99: # 物体大小:
        print('Skip: Generation mask fails')
        return None

    total_edited_images = []
    if mask_mode == 'count':
        edited_image = image_pil
        if mask_pil.shape[0] == 1 or remove_number == 1:
            print('Skip: mask_pil.shape[0] == 1 or remove_number == 1')
            return None
        for i in range(min(mask_pil.shape[0], remove_number)):
            mask_path = edited_image_path_pre.replace("edited_img", "mask") + f'_{i}.png'
            mask = mask_pil[i][0].cpu().numpy()
            mask = Image.fromarray(mask)
            mask.save(mask_path)
            cv2_mask_image = cv2.imread(mask_path)  
            maskimage_dilate = cv2.dilate(cv2_mask_image, np.ones((15, 15), np.uint8))    
            maskimage_dilate = Image.fromarray(cv2.cvtColor(maskimage_dilate,cv2.COLOR_BGR2GRAY))
            negative_prompt = f'{edited_object}, man, woman, peroson, objects, text, watermark, bad quality'
            edited_image = inpaint_pipe(prompt="", negative_prompt=negative_prompt, image=edited_image,
                                        mask_image=maskimage_dilate).images[0]
            instruct_word = random.choice(['Remove', 'Delete', 'Erase'])
            cur_instruction = f'{instruct_word} {i+1} of {edited_object} in the image.'
            cur_edited_image_path = edited_image_path_pre + f'_{i}.png'
            cur_data = {"edit": cur_instruction, "edit object": edited_object, "output": output, "input": input,
                        "edit_type": "counting", "image_file": init_image_path, "edited_file": cur_edited_image_path}
            edited_image = edited_image.resize(original_size, resample=Image.Resampling.LANCZOS)
            edited_image.save(cur_edited_image_path)
            total_edited_images.append(cur_data)
        return total_edited_images
    else:
        mask_path = edited_image_path_pre.replace("edited_img", "mask") + f'.png'
        mask_pil.save(mask_path)
        mask_bbox_path = edited_image_path_pre.replace("edited_img", "mask") + f'_bbox.png'
        mask_bbox.save(mask_bbox_path)
        cv2_mask_image = cv2.imread(mask_path)  
        if union_region < 0.33:
            maskimage_dilate = cv2.dilate(cv2_mask_image, np.ones((8, 8), np.uint8))
        elif union_region < 0.66:
            maskimage_dilate = cv2.dilate(cv2_mask_image, np.ones((15, 15), np.uint8))
        else:
            maskimage_dilate = cv2.dilate(cv2_mask_image, np.ones((20, 20), np.uint8))
        maskimage_dilate = Image.fromarray(cv2.cvtColor(maskimage_dilate,cv2.COLOR_BGR2GRAY))
        negative_prompt = f'{edited_object}, man, woman, peroson, objects, text, watermark, bad quality'
        total_edited_images = []
        for i in range(1):
            edited_image = inpaint_pipe(prompt="", negative_prompt=negative_prompt, image=image_pil, mask_image=maskimage_dilate).images[0]
            edit_mask_pil, _, _, _ = maskgeneration(det_model, sam_model, edited_image, prompt, mask_mode=mask_mode,
                                                    target_object=edited_object)
            if edit_mask_pil is None: 
                edited_image = edited_image.resize(original_size, resample=Image.Resampling.LANCZOS)
                cur_edited_image_path = edited_image_path_pre+f'_{i}.png'
                cur_data = {"edit": instruction, "edit object": edited_object, "output": output, "input": input,
                            "edit_type": "remove", "image_file": init_image_path, "edited_file": cur_edited_image_path}
                edited_image.save(cur_edited_image_path)
                total_edited_images.append(cur_data)
                break
            else:
                intersection_mask = np.asarray(mask_pil) & np.asarray(edit_mask_pil)
                num_inter_ones = np.sum(intersection_mask == 1)
                num_original_ones = np.sum(np.asarray(mask_pil)== 1)
                percentage = (num_inter_ones / num_original_ones)
                if percentage < 0.2:
                    edited_image = edited_image.resize(original_size, resample=Image.Resampling.LANCZOS)
                    cur_edited_image_path = edited_image_path_pre+f'_{i}.png'
                    cur_data = {"edit": instruction, "edit object": edited_object, "output": output, "input": input,
                                "edit_type": "remove", "image_file": init_image_path, "edited_file": cur_edited_image_path}
                    edited_image.save(cur_edited_image_path)
                    total_edited_images.append(cur_data)
                    break
        return total_edited_images

def replace(det_model, sam_model, inpaint_pipe, input, edited_object, new_object, output, init_image_path,
            instruction,  edited_image_path_pre, sdxl_base=None):
    if edited_object in ['people', 'man', 'woman', 'boy', 'girl', 'person'] or new_object in ['people', 'man', 'woman', 'boy', 'girl', 'person']:
        return None
    if not os.path.exists(init_image_path): 
        init_image = sdxl_pipe(sdxl_base, input)
        init_image.save(init_image_path)
    total_objects = generate_tags(input)['nouns']
    if edited_object not in total_objects:
        total_objects.append(edited_object)
    prompt = ', '.join(total_objects)
    original_size = Image.open(init_image_path).convert("RGB").size
    mask_pil, image_pil, mask_bbox, union_region = maskgeneration(det_model, sam_model, init_image_path, prompt, mask_mode='max', target_object=edited_object)
    if mask_pil is None:
        print('Skip: mask_pil is None')
        return None
    if union_region < 0.05:
        print('Skip: union_region is too small')
        return None
    if union_region > 0.95:
        print('Skip: union_region is too big')
        return None
    # artifact correction
    mask_path =  edited_image_path_pre.replace("edited_img", "mask") + f'.jpg'
    mask_bbox_path = edited_image_path_pre.replace("edited_img", "mask") + f'_bbox.jpg'
    mask_bbox.save(mask_bbox_path)
    mask_pil.save(mask_path)

    cv2_mask_image = cv2.imread(mask_path)
    # if union_region < 0.33:
    #     maskimage_dilate = cv2.dilate(cv2_mask_image, np.ones((50, 50), np.uint8))
    # elif union_region < 0.66:
    #     maskimage_dilate = cv2.dilate(cv2_mask_image, np.ones((20, 20), np.uint8))
    # else:
    maskimage_dilate = cv2.dilate(cv2_mask_image, np.ones((30, 30), np.uint8))
    maskimage_dilate_extend = cv2.dilate(cv2_mask_image, np.ones((20, 20), np.uint8))

    maskimage_dilate = Image.fromarray(cv2.cvtColor(maskimage_dilate,cv2.COLOR_BGR2GRAY))
    maskimage_dilate_extend = Image.fromarray(cv2.cvtColor(maskimage_dilate_extend,cv2.COLOR_BGR2GRAY))
    target_objects = generate_tags(output)['nouns']
    if new_object not in target_objects:
        target_objects.append(new_object)
    for obj in total_objects:
        if obj not in target_objects:
            target_objects.append(obj)
    edit_prompt = ', '.join(target_objects)
    total_edited_images = []
    for i in range(2):
        negative_prompt = f"{edited_object}, bad quality, text, watermark"
        edited_image = inpaint_pipe(prompt=new_object, negative_prompt=negative_prompt, image=image_pil, mask_image=maskimage_dilate).images[0]
        edited_image = inpaint_pipe(prompt=new_object, negative_prompt="bad quality", image=edited_image, mask_image=maskimage_dilate_extend).images[0]
        edit_mask_pil, _, edit_mask_bbox, _ = maskgeneration(det_model, sam_model, edited_image, edit_prompt, mask_mode='max', target_object=new_object)
        if edit_mask_pil is not None:
            edited_image = edited_image.resize(original_size, resample=Image.Resampling.LANCZOS)
            cur_edited_image_path =  edited_image_path_pre + f'.png'
            edited_image.save(cur_edited_image_path)
            
            cur_data = {"edit": instruction, "edit object": edited_object, "new object": new_object, "output": output,
                        "input": input, "edit_type": "replace", "image_file": init_image_path, "edited_file": cur_edited_image_path}
            total_edited_images.append(cur_data)
            break
    if len(total_edited_images) == 0:
        os.remove(mask_path)
    return total_edited_images

def background_change(det_model, sam_model, inpaint_pipe, input, new_background,
                      init_image_path, edited_image_path_pre, instruction, foreground_object,
                      sdxl_base=None):
    if not os.path.exists(init_image_path): 
        init_image = sdxl_pipe(sdxl_base, input)
        init_image.save(init_image_path)
    # total_objects = generate_tags(output)['nouns']
    back_mask_pil = None
    prompt = ''
    if foreground_object is not None and len(foreground_object) > 0:
        # foreground_object = total_objects[:-1]
        prompt = ', '.join(foreground_object)
        prompt = prompt + ', people, man, woman, boy, girl, person'
        fore_mask_pil, _, _, fore_union_region = maskgeneration(det_model, sam_model, init_image_path, prompt, box_threshold=0.2, text_threshold=0.2, mask_mode='merge')
        face_mask_pil = None
        for obj in foreground_object:
            if obj in ['people', 'man', 'woman', 'boy', 'girl', 'person'] or is_human_variant(obj):
                prompt = prompt + ', face, head'
                face_mask_pil, _, _, _ = maskgeneration(det_model, sam_model, init_image_path, prompt,
                                                            mask_mode='merge', box_threshold=0.2, text_threshold=0.2,
                                                            target_object=["face", "head"])
                break
        if fore_mask_pil is not None and face_mask_pil is not None:
            fore_intersection_mask = np.asarray(fore_mask_pil) | np.asarray(face_mask_pil)
        elif fore_mask_pil is not None:
            fore_intersection_mask = np.asarray(fore_mask_pil)
        elif face_mask_pil is not None:
            fore_intersection_mask = np.asarray(face_mask_pil)
        else:
            fore_intersection_mask = None
        if fore_intersection_mask is not None:
            fore_intersection_mask = Image.fromarray(fore_intersection_mask)
            temp_mask = edited_image_path_pre.replace("edited_img", "mask") + f'_foreground.png'
            fore_intersection_mask.save(temp_mask)
            fore_mask_image = cv2.imread(temp_mask)
            fore_dilate = cv2.dilate(fore_mask_image, np.ones((30, 30), np.uint8))
            fore_dilate = cv2.GaussianBlur(fore_mask_image, (5, 5), 0) 
            fore_dilate = Image.fromarray(cv2.cvtColor(fore_dilate,cv2.COLOR_BGR2GRAY))
            os.remove(temp_mask)
            back_mask_pil = Image.fromarray(np.where(np.asarray(fore_dilate)==0, True, False))

    image_pil = Image.open(init_image_path).convert("RGB").resize((512, 512), resample=Image.Resampling.LANCZOS)
    original_size = Image.open(init_image_path).convert("RGB").size
    if back_mask_pil is None:
        print('Skip: mask_pil error')
        return None
    # elif fore_union_region is None or fore_union_region > 0.9:
        # print('Skip: back_mask_pil is not None but fore_union_region > 0.9')
        # return None
    intersection_mask = np.asarray(back_mask_pil)
    if np.sum(intersection_mask > 0)/intersection_mask.size < 0.05:
        print('Skip: np.sum(intersection_mask > 0)/intersection_mask.size < 0.1')
        return None

    intersection_mask = Image.fromarray(intersection_mask)
    # artifact correction
    mask_path = edited_image_path_pre.replace("edited_img", "mask") + f'.png'

    total_edited_images = []
    for i in range(1):
        negative_prompt = f"bad anatomy, bad hands, person, man, woman, people, ugly, bad anatomy, bad proportions, bad quality, text, watermark, low quality, user name, bad quality"       
        edited_image = inpaint_pipe(prompt=new_background, negative_prompt=negative_prompt, image=image_pil, mask_image=intersection_mask).images[0]
        # edited_image = cropimage2image(edited_image, maskimage_dilate, image_pil, scale=1)
        edited_image = edited_image.resize(original_size, resample=Image.Resampling.LANCZOS)
        cur_edited_image_path = edited_image_path_pre+f'_{i}.png'
        intersection_mask.resize(original_size, resample=Image.Resampling.LANCZOS).save(mask_path)
        edited_image.save(cur_edited_image_path)
        # image_pil.save(edited_image_path_pre.replace("edited_img", "input_img") + f'.png')
        cur_data = {"edit": instruction, "new background": new_background, "output": None, "input": input,
                    "edit_type": "background_change", "image_file": init_image_path, "foreground": foreground_object, "edited_file": cur_edited_image_path}
        total_edited_images.append(cur_data)
    return total_edited_images

def resize_movement(det_model, sam_model, inpaint_pipe, input, edited_object,
                    init_image_path, edited_image_path_pre, instruction_type, sdxl_base=None, sdxl_refiner=None):
    # generate init image: x
    if not os.path.exists(init_image_path):
        # init_image = sdxl_pipe(sdxl_base, sdxl_refiner, input)
        init_image = sdxl_pipe(sdxl_base, input)
        init_image.save(init_image_path)

    original_size = Image.open(init_image_path).convert("RGB").size
    total_objects = generate_tags(input)['nouns']
    if edited_object not in total_objects:
        total_objects.append(edited_object)

    prompt = ', '.join(total_objects)
    mask_pil, image_pil, mask_bbox, union_region = maskgeneration(det_model, sam_model, init_image_path, prompt, target_object=edited_object)

    if mask_pil is None:
        print('Skip: mask is none')
        return None

    mask_array = np.array(mask_pil)
    coords = np.argwhere(mask_array > 0)
    min_y, min_x = np.min(coords, axis=0)
    max_y, max_x = np.max(coords, axis=0)
    mask_size = [max_x-min_x, max_y-min_y]
    total_edited_images = []

    mask_path = edited_image_path_pre.replace("edited_img", "mask") + f'.png'
    mask_pil.save(mask_path)
    cv2_mask_image = cv2.imread(mask_path)

    if union_region < 0.33:
        maskimage_dilate = cv2.dilate(cv2_mask_image, np.ones((8, 8), np.uint8))
    elif union_region < 0.66:
        maskimage_dilate = cv2.dilate(cv2_mask_image, np.ones((15, 15), np.uint8))
    else:
        maskimage_dilate = cv2.dilate(cv2_mask_image, np.ones((20, 20), np.uint8))    
    maskimage_dilate = Image.fromarray(cv2.cvtColor(maskimage_dilate,cv2.COLOR_BGR2GRAY))
    negative_prompt = f'{edited_object}, text, watermark, low quality'
    edited_image = inpaint_pipe(prompt="", negative_prompt=negative_prompt, image=image_pil, mask_image=maskimage_dilate).images[0]
    
    edit_mask_pil, _, _, _ = maskgeneration(det_model, sam_model, edited_image, prompt, target_object=edited_object)
    if edit_mask_pil is None:
        # now to add movement/resize object
        if instruction_type == 'movement':
            scale = 1
            if int(1-min_x) != (image_pil.size[0]-1-int(mask_size[0]*scale)-int(min_x)):
                delta_x = random.randint(int(1-min_x), image_pil.size[0]-1-int(mask_size[0]*scale)-int(min_x))
            # delta_y = random.randint(int(1-max_y+int(mask_size[1]*scale)), image_pil.size[1]-1-int(max_y))
            # delta_x = 100
            else:
                delta_x = int(1-min_x)
            x = max(1, min_x + delta_x)
            # y = min(1, max_y - int(mask_size[1]*scale) + delta_y)
            y = max_y - int(mask_size[1]*scale)
        else:
            if int((image_pil.size[0]-min_x)/mask_size[0]) < 1.5 or int((image_pil.size[1]-min_y)/mask_size[1]) < 1.5:
                scale = 0.5
            else:
                scale = 1.5
            x = min_x
            y = max_y - int(mask_size[1]*scale)
        result_image_pil = resize_cropimage2image(image_pil, mask_pil, edited_image, x, y, scale)
        edited_image = result_image_pil.resize(original_size, resample=Image.Resampling.LANCZOS)
        i = 0
        cur_edited_image_path = edited_image_path_pre+f'_{i}.png'
        if scale == 1:
            instruction_word = random.choice(['move', 'shift'])
            cur_data = {"edit": f"{instruction_word} the {edited_object} in the image", "edit object": edited_object,
                        "output": None, "input": input, "edit_type": "movement", "image_file": init_image_path,
                        "edited_file": cur_edited_image_path}
        elif scale == 1.5:
            instruction_word = random.choice(['zoom in', 'enlarge', 'amplify'])
            cur_data = {"edit": f"{instruction_word} the {edited_object} in the image", "edit object": edited_object,
                        "output": None, "input": input, "edit_type": "resize", "image_file": init_image_path,
                        "edited_file": cur_edited_image_path}
        else:
            instruction_word = random.choice(['zoom out', 'minify'])
            cur_data = {"edit": f"{instruction_word} the {edited_object} in the image", "edit object": edited_object,
                        "output": None, "input": input, "edit_type": "resize", "image_file": init_image_path,
                        "edited_file": cur_edited_image_path}
        cur_data['scale'] = int(scale)
        cur_data['x'] = int(x)
        cur_data['y'] = int(y)
        edited_image.save(cur_edited_image_path)
        total_edited_images.append(cur_data)
    return total_edited_images

# def file_exists(file_path):
#     return os.path.exists(file_path)
def read_json_file(file_path):
    if os.path.exists(file_path):
        with open(file_path, 'r') as file:
            return json.load(file)
    else:
        print(f"File {file_path} does not exist.")
        return []
def parse_args():
    parser = argparse.ArgumentParser(description="Editing Pipeline")
    parser.add_argument("--groundingdino-config-file", required=True, help="path to groundingdino config file.")
    parser.add_argument("--groundingdino-ckpt", required=True, help="path to groundingdino model file")
    parser.add_argument("--sam-ckpt", required=True, help="path to sam model file")
    parser.add_argument("--sdxl-ckpt", required=True, help="path to sdxl model file")
    parser.add_argument("--inpaint-ckpt", required=True, help="path to inpaint model file")
    parser.add_argument("--instruction-path", required=True, help="path to instruction path")
    parser.add_argument("--instruction-type", required=True, help="specify the instruction type.")
    parser.add_argument("--json-path", default=None)
    parser.add_argument("--image-path", default=None)
    parser.add_argument("--start-idx", type=str, default='-1', help="specify the experiment id.")
    parser.add_argument("--end-idx", type=str, default='-1', help="specify the experiment id.")
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
    instruction_path = f'{args.instruction_path}/{args.instruction_type}'
    final_edit_results_path = f'{instruction_path}/final_edit_results_{args.start_idx}_{args.end_idx}.json'
    final_edited_results = read_json_file(final_edit_results_path)
    
    edit_success_path = f'{instruction_path}/edit_success_{args.start_idx}_{args.end_idx}.json'
    success_data = read_json_file(edit_success_path)

    edit_failure_path = f'{instruction_path}/edit_failure_{args.start_idx}_{args.end_idx}.json'
    failure_data = read_json_file(edit_failure_path)

    os.makedirs(f'{args.instruction_path}/{args.instruction_type}/mask', exist_ok=True)
    os.makedirs(f'{args.instruction_path}/{args.instruction_type}/input_img', exist_ok=True)
    os.makedirs(f'{args.instruction_path}/{args.instruction_type}/edited_img', exist_ok=True)

    sdxl_base, inpaint_pipe, det_model, sam_model = load_tool_model(args)
    valid_number = 0

    iter = 0
    if args.start_idx != '-1' and args.end_idx != '-1':
        print(args.start_idx, args.end_idx)
        st_idx = int(args.start_idx)
        ed_idx = int(args.end_idx)
        print(st_idx, ed_idx)
        edit_instruction_data = edit_instruction_data[st_idx:ed_idx]
    print(len(edit_instruction_data))
    for data in tqdm(edit_instruction_data):
        try:
            iter += 1
            instruction_type = data['edit_type']
            if 'input' not in data or 'output' not in data:
                print(f'Skip: input or output in data')
                continue

            if instruction_type in ["add", "remove", "counting", "background_change", "replace", "resize", "movement"]:
                if instruction_type in ["add", "remove", "counting"] \
                        and ('edited object' not in data or data['edited object'] not in data['edit'] or 'background' in data['edit']):
                    print(f'Skip: no edited object in data')
                    continue
                elif instruction_type in ["resize", "movement"] and 'edited object' not in data:
                    print(f'Skip: no edited object in data')
                    continue
                elif instruction_type== "replace" and ('edited object' not in data or 'new object' not in data or data['edited object'] not in data['edit'] or 'background' in data['edit']):
                    print(f'Skip: no edited object in data for replace type')
                    continue
                elif instruction_type=="background_change" and 'new background' not in data:
                    print(f'Skip: background change edit should have new background')
                    continue

                if instruction_type in ["add", "remove", "resize", "movement"]:
                    input, edited_object, output, init_image_path, instruction, edited_image_path_pre = \
                    return_parameters(data, args, init_image_root=args.image_path)
                elif instruction_type == "counting":
                    input, edited_object, output, init_image_path, instruction, edited_image_path_pre, remove_number = \
                        return_parameters(data, args, init_image_root=args.image_path)
                elif instruction_type == "background_change":
                    input, output, init_image_path, instruction, edited_image_path_pre = return_parameters(data, args, init_image_root=args.image_path)
                elif instruction_type == "replace":
                    input, edited_object, new_object, output, init_image_path, instruction,  edited_image_path_pre = \
                        return_parameters(data, args, init_image_root=args.image_path)
                else:
                    raise NotImplementedError

                # continue generation for NVCC
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
                    print(f'Skip: output image exist')
                    valid_number += 1
                    if instruction_type == "background_change":
                        edit_image = {"edit": instruction, "edit object": None, "output": output, "input": input, "edit_type": instruction_type, "image_file": init_image_path, "edited_file": save_edited_image_path}
                        final_edited_results.append(edit_image)
                        success_data.append(data)
                        continue
                    else:
                        edit_image = {"edit": instruction, "edit object": edited_object, "output": output, "input": input, "edit_type": instruction_type, "image_file": init_image_path, "edited_file": save_edited_image_path}
                        final_edited_results.append(edit_image)
                        success_data.append(data)
                        continue

                if instruction_type == "add":
                    instruction_data = add(det_model=det_model, sam_model=sam_model, inpaint_pipe=inpaint_pipe,
                                    edited_object=edited_object, output=output,
                                    instruction=instruction, edited_image_path_pre=edited_image_path_pre,
                                    sdxl_base=sdxl_base)
                elif instruction_type == "remove":
                    instruction_data = remove(det_model=det_model, sam_model=sam_model, inpaint_pipe=inpaint_pipe,
                                            input=input, output=output, edited_object=edited_object, init_image_path=init_image_path,
                                            instruction=instruction, edited_image_path_pre=edited_image_path_pre,
                                            sdxl_base=sdxl_base, mask_mode='merge')
                elif instruction_type == "counting":
                    instruction_data = remove(det_model=det_model, sam_model=sam_model, inpaint_pipe=inpaint_pipe,
                                            input=input, output=output, edited_object=edited_object, init_image_path=init_image_path,
                                            instruction=instruction, edited_image_path_pre=edited_image_path_pre,
                                            sdxl_base=sdxl_base, mask_mode= 'count', remove_number=remove_number)
                elif instruction_type == "background_change":
                    instruction_data = background_change(det_model=det_model, sam_model=sam_model, inpaint_pipe=inpaint_pipe,
                                                        input=input, new_background=data['new background'],
                                                        init_image_path=init_image_path, edited_image_path_pre=edited_image_path_pre,
                                                        instruction=instruction,
                                                        foreground_object=data['foreground'] if 'foreground' in data.keys() else None,
                                                        sdxl_base=sdxl_base)
                elif instruction_type == "replace":
                    instruction_data = replace(det_model=det_model, sam_model=sam_model, inpaint_pipe=inpaint_pipe,
                                            input=input, edited_object=edited_object,
                                            new_object=new_object, output=output, init_image_path=init_image_path,
                                            instruction=instruction,  edited_image_path_pre=edited_image_path_pre,
                                            sdxl_base=sdxl_base)
                elif instruction_type in ["resize", "movement"]:
                    instruction_data = resize_movement(det_model=det_model, sam_model=sam_model, inpaint_pipe=inpaint_pipe,
                                                    input=input, edited_object=edited_object,
                                                    init_image_path=init_image_path, edited_image_path_pre=edited_image_path_pre,
                                                    instruction_type=instruction_type, sdxl_base=sdxl_base)
                else:
                    raise NotImplementedError
            else:
                raise NotImplementedError

            if instruction_data is not None and len(instruction_data) > 0:
                valid_number += 1
                final_edited_results.extend(instruction_data)
                success_data.append(data)
            else:
                generated_image_file = 'flux_' + data['image_file'].replace('.jpg', '.png')
                init_image_path = f"/flux_coco_images/{generated_image_file}"
                if instruction_type == "remove":
                    update_instruction_data = remove(det_model=det_model, sam_model=sam_model, inpaint_pipe=inpaint_pipe,
                                            input=input, output=output, edited_object=edited_object, init_image_path=init_image_path,
                                            instruction=instruction, edited_image_path_pre=edited_image_path_pre,
                                            sdxl_base=sdxl_base, mask_mode='merge')
                elif instruction_type == "counting":
                    update_instruction_data = remove(det_model=det_model, sam_model=sam_model, inpaint_pipe=inpaint_pipe,
                                            input=input, output=output, edited_object=edited_object, init_image_path=init_image_path,
                                            instruction=instruction, edited_image_path_pre=edited_image_path_pre,
                                            sdxl_base=sdxl_base, mask_mode= 'count', remove_number=remove_number)
                elif instruction_type == "background_change":
                    update_instruction_data = background_change(det_model=det_model, sam_model=sam_model, inpaint_pipe=inpaint_pipe,
                                                        input=input, new_background=data['new background'],
                                                        init_image_path=init_image_path, edited_image_path_pre=edited_image_path_pre,
                                                        instruction=instruction,
                                                        foreground_object=data['foreground'] if 'foreground' in data.keys() else None,
                                                        sdxl_base=sdxl_base)
                elif instruction_type == "replace":
                    update_instruction_data = replace(det_model=det_model, sam_model=sam_model, inpaint_pipe=inpaint_pipe,
                                            input=input, edited_object=edited_object,
                                            new_object=new_object, output=output, init_image_path=init_image_path,
                                            instruction=instruction,  edited_image_path_pre=edited_image_path_pre,
                                            sdxl_base=sdxl_base)
                elif instruction_type in ["resize", "movement"]:
                    update_instruction_data = resize_movement(det_model=det_model, sam_model=sam_model, inpaint_pipe=inpaint_pipe,
                                                    input=input, edited_object=edited_object,
                                                    init_image_path=init_image_path, edited_image_path_pre=edited_image_path_pre,
                                                    instruction_type=instruction_type, sdxl_base=sdxl_base)

                if update_instruction_data is not None and len(update_instruction_data) > 0:
                    valid_number += 1
                    final_edited_results.extend(update_instruction_data)
                    success_data.append(data)
                else:
                    failure_data.append(data)
        except Exception as e:
            print(e)
            continue
        

    print(f"valid editing insturction data: {valid_number}")
    with open(f'{args.instruction_path}/{args.instruction_type}/final_edit_results_{args.start_idx}_{args.end_idx}.json', 'w') as results_file:
        json.dump(final_edited_results, results_file, indent=4)
    with open(f'{args.instruction_path}/{args.instruction_type}/edit_success_{args.start_idx}_{args.end_idx}.json', 'w') as success_file:
        json.dump(success_data, success_file, indent=4)
    with open(f'{args.instruction_path}/{args.instruction_type}/edit_failure_{args.start_idx}_{args.end_idx}.json', 'w') as failure_file:
        json.dump(failure_data, failure_file, indent=4)


