import os
import numpy as np
from skimage import morphology
import mediapipe as mp
import warnings
import torch
import cv2
from torchvision.transforms import Compose
from depth_anything_v2.dpt import DepthAnythingV2
import matplotlib
import glob
from DPT.util import io
from DPT.dpt.models import DPTDepthModel
from DPT.dpt.transforms import Resize, NormalizeImage, PrepareForNet
from termcolor import cprint
from checkpoints.visual_reference.uniformer.mmseg.datasets.pipelines import Compose
from checkpoints.visual_reference.uniformer.mmseg.apis import init_segmentor, inference_segmentor, show_result_pyplot
from checkpoints.visual_reference.uniformer.mmseg.core.evaluation import get_palette
from checkpoints.visual_reference.hed import HEDdetector

warnings.filterwarnings("ignore")

# os.environ['CUDA_VISIBLE_DEVICES'] = '6' # todo: 这个之后要移到外面去

# only work for human
def img2openpose(image_path, output_path):

    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Initialize MediaPipe Pose with GPU support
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=True, model_complexity=1, min_detection_confidence=0.5)

    # Process the image to detect pose
    results = pose.process(image_rgb)

    # Draw the pose annotations on the image
    mp_drawing = mp.solutions.drawing_utils
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            image_rgb, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2),
            mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2)
        )

    # Convert the annotated image back to BGR for saving
    annotated_image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

    # Save the annotated image
    cv2.imwrite(output_path, annotated_image_bgr)

def img2sketch(image_path, output_path):
    # Load the image
    img = cv2.imread(image_path)
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Apply Gaussian blur
    gaussian_blur = cv2.GaussianBlur(gray, (3, 3), 0)
    # Apply edge detection
    edges = cv2.Canny(gaussian_blur, 50, 150)
    # Invert the colors of the edges
    edges = cv2.bitwise_not(edges)
    # Save the result
    cv2.imwrite(output_path, edges)

def run_depth(input_image, output_image, model):
    """Run MonoDepthNN to compute depth maps.

    Args:
        input_path (str): path to input image
        output_path (str): path to output image
        model_path (str): path to saved model
    """

    # select device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    net_w = net_h = 384
    normalization = NormalizeImage(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    transform = Compose(
        [
            Resize(
                net_w,
                net_h,
                resize_target=None,
                keep_aspect_ratio=True,
                ensure_multiple_of=32,
                resize_method="minimal",
                image_interpolation_method=cv2.INTER_CUBIC,
            ),
            normalization,
            PrepareForNet(),
        ]
    )
    model.eval()

    if device == torch.device("cuda"):
        model = model.to(memory_format=torch.channels_last)
        model = model.half()

    model.to(device)
    img = io.read_image(input_image)

    img_input = transform({"image": img})["image"]

    # compute
    with torch.no_grad():
        sample = torch.from_numpy(img_input).to(device).unsqueeze(0)

        if device == torch.device("cuda"):
            sample = sample.to(memory_format=torch.channels_last)
            sample = sample.half()

        prediction = model.forward(sample)
        prediction = (
            torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=img.shape[:2],
                mode="bicubic",
                align_corners=False,
            )
            .squeeze()
            .cpu()
            .numpy()
        )
    cprint(output_image, 'red')
    io.write_depth(output_image, prediction, bits=2, absolute_depth=False)

# def img2depth(image_path, output_path):
#     device = 'cuda'
#     dpt_model_path = './checkpoints/material_transfer/dpt_hybrid-midas-501f0c75.pt'
#     dpt_model = DPTDepthModel(
#         path=dpt_model_path,
#         backbone="vitb_rn50_384",
#         non_negative=True,
#         enable_attention_hooks=False,
#     )
#     dpt_model = dpt_model.to(device)
#     run_depth(image_path, output_path, dpt_model)

def img2depth(image_path, output_path):
    '''
    change to depth anything V2
    '''
    DEVICE = 'cuda'
    depth_anything = DepthAnythingV2(encoder='vitl', features=256, out_channels=[256, 512, 1024, 1024])
    depth_anything.load_state_dict(
        torch.load(f'./checkpoints/visual_reference/depth_anything_v2/depth_anything_v2_vitl.pth',
                   map_location='cpu'))
    depth_anything = depth_anything.to(DEVICE).eval()

    if os.path.isfile(image_path):
        if image_path.endswith('txt'):
            with open(image_path, 'r') as f:
                filenames = f.read().splitlines()
        else:
            filenames = [image_path]
    else:
        filenames = glob.glob(os.path.join(image_path, '**/*'), recursive=True)

    cmap = matplotlib.colormaps.get_cmap('Spectral_r')

    for k, filename in enumerate(filenames):
        raw_image = cv2.imread(filename)
        depth = depth_anything.infer_image(raw_image)
        depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
        depth = depth.astype(np.uint8)
        depth = (cmap(depth)[:, :, :3] * 255)[:, :, ::-1].astype(np.uint8)
        cv2.imwrite(output_path, depth)


def img2action(image_path, output_path):
    img = cv2.imread(image_path,0)
    _,binary = cv2.threshold(img,200,255,cv2.THRESH_BINARY_INV)

    binary[binary==255] = 1
    skel, distance =morphology.medial_axis(binary, return_distance=True)
    dist_on_skel = distance * skel
    dist_on_skel = dist_on_skel.astype(np.uint8)*255
    cv2.imwrite(output_path,dist_on_skel)

def img2seg(image_path, output_path, model):
    # Load the image
    img = cv2.imread(image_path)
    result = inference_segmentor(model, img)
    res_img = show_result_pyplot(model, img, result, get_palette('ade'), opacity=1)
    # Save the result
    cv2.imwrite(output_path, res_img)

if __name__ == '__main__':
    image_path = './edit_generated_datasets/example_dataset/other_visual/input_img/test_input.png'

    # segment
    seg_model = init_segmentor(config='./checkpoints/visual_reference/uniformer/seg_config.py', device='cuda',
                               checkpoint='./checkpoints/visual_reference/annotator/ckpts/upernet_global_small.pth')
    img2seg('./edit_generated_datasets/example_dataset/other_visual/input_img/house.png',
            './edit_generated_datasets/example_dataset/other_visual/test_seg.png', seg_model)
    # scribble
    hed_model = HEDdetector(path='./checkpoints/visual_reference/ControlNetHED.pth')
    hed_model(image_path='./edit_generated_datasets/example_dataset/other_visual/input_img/bag.png',
              output_path='./edit_generated_datasets/example_dataset/other_visual/test_scr.png')

    img2sketch(image_path, './edit_generated_datasets/example_dataset/other_visual/test_sketch.png')
    img2depth(image_path, './edit_generated_datasets/example_dataset/other_visual/test_depth.png')

    # zw@zju: poor performance
    # 就算是人的不是全身也不行
    # img2openpose(image_path, './edit_generated_datasets/example_dataset/other_visual/test_openpose.png')
    # 这很烂
    # img2action(image_path, './edit_generated_datasets/example_dataset/other_visual/test_action.png')
