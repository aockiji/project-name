"""
segmentation.py

Wraps 4 segmentation methods behind a common interface:
- Felzenszwalb (graph-based)
- Normalized Cut (graph-based)
- DeepLabV3 (deep learning baseline)
- FCN (deep learning baseline)
will add other methods in a bit

Each function takes a numpy image (RGB, uint8) and returns:
    (output_image, runtime_seconds)
where output_image is a numpy image ready to save with PIL/cv2.
"""

import time
import numpy as np
from PIL import Image

from skimage.segmentation import felzenszwalb, slic, mark_boundaries
from skimage import graph

import torch
from torchvision import models, transforms

MAX_SIZE = 300  # longest side in pixels


def resize_image(image: np.ndarray, max_size: int = MAX_SIZE) -> np.ndarray:
    """Resize so the longest side is max_size, keeping aspect ratio."""
    h, w = image.shape[:2]
    scale = max_size / max(h, w)
    if scale >= 1:
        return image  # don't upscale small images
    new_w, new_h = int(w * scale), int(h * scale)
    pil_img = Image.fromarray(image).resize((new_w, new_h), Image.LANCZOS)
    return np.array(pil_img)


# graphbased methods

def run_felzenszwalb(image: np.ndarray):
    """Felzenszwalb-Huttenlocher graph-based segmentation."""
    start = time.time()
    segments = felzenszwalb(image, scale=1000, sigma=0.5, min_size=50) # scale=k, sigma for gaussian smoothing before segmentation, min_size for extra regions
    output = mark_boundaries(image, segments) # mark the boundaries between the componenets
    output = (output * 255).astype(np.uint8) # mark_boudnaries retursn image w pixel vals as floaitng point betweeen 0 to 1 so this converts back to 8bit int w range 0 to 255
    runtime = time.time() - start
    return output, runtime


def run_normalized_cut(image: np.ndarray):
    """Normalized Cut segmentation, built on top of SLIC superpixels."""
    start = time.time()

    superpixels = slic(image, n_segments=200, compactness=10, start_label=1)
    rag = graph.rag_mean_color(image, superpixels, mode='similarity')

    segments = graph.cut_normalized(superpixels, rag)
    output = mark_boundaries(image, segments)
    output = (output * 255).astype(np.uint8)
    runtime = time.time() - start
    return output, runtime


# deep learning baselines

_deeplab_model = None
_fcn_model = None

_preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                          std=[0.229, 0.224, 0.225]),
])

_PALETTE = np.random.RandomState(42).randint(0, 255, size=(21, 3), dtype=np.uint8)
_PALETTE[0] = [0, 0, 0]  # background = black


def _get_deeplab():
    global _deeplab_model
    if _deeplab_model is None:
        _deeplab_model = models.segmentation.deeplabv3_resnet50(pretrained=True)
        _deeplab_model.eval()
    return _deeplab_model


def _get_fcn():
    global _fcn_model
    if _fcn_model is None:
        _fcn_model = models.segmentation.fcn_resnet50(pretrained=True)
        _fcn_model.eval()
    return _fcn_model


def _run_dl_model(image: np.ndarray, model):
    input_tensor = _preprocess(image).unsqueeze(0)
    with torch.no_grad():
        output = model(input_tensor)['out'][0]
    class_map = output.argmax(0).byte().numpy()
    color_map = _PALETTE[class_map]
    return color_map


def run_deeplab(image: np.ndarray):
    start = time.time()
    output = _run_dl_model(image, _get_deeplab())
    runtime = time.time() - start
    return output, runtime


def run_fcn(image: np.ndarray):
    start = time.time()
    output = _run_dl_model(image, _get_fcn())
    runtime = time.time() - start
    return output, runtime


def run_all(image: np.ndarray):
    """
    Runs all 4 algos on the given image (already resized).
    Returns a dict: { name: {"image": np.ndarray, "runtime": float} }
    """
    image = resize_image(image)

    results = {}

    for name, func in [
        ("Felzenszwalb-Huttenlocher", run_felzenszwalb),
        ("Normalized Cut", run_normalized_cut),
        ("DeepLabV3", run_deeplab),
        ("FCN", run_fcn),
    ]:
        output, runtime = func(image)
        results[name] = {"image": output, "runtime": round(runtime, 3)}

    return results