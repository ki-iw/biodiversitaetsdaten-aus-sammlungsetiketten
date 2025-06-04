from pathlib import Path

import cv2
from prefect import task

from kiebids import config, pipeline_config, run_id
from kiebids.utils import debug_writer, get_kiebids_logger, resize

module = __name__.split(".")[-1]

debug_path = (
    "" if config.mode != "debug" else f"{config['debug_path']}/{module}/{run_id}"
)
module_config = pipeline_config[module]


@task(name=module)
@debug_writer(debug_path, module=module)
def preprocessing(current_image_name):
    image_path = Path(config.image_path) / current_image_name
    logger = get_kiebids_logger(module)
    logger.info("Preprocessing image: %s", image_path)
    image = cv2.imread(image_path)

    image = resize(image, module_config.max_image_dimension)

    if module_config["gray"]:
        image = gray(image)

    if module_config["smooth"]:
        image = smooth(image)

    if module_config["threshold"]:
        image = threshold(image)

    if module_config["denoise"]:
        image = denoise(image)

    if module_config["contrast"]:
        image = contrast(image)

    return image


def gray(image):
    """Converts an image to grayscale"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Convert back BGR to keep 3 color channels
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def smooth(image):
    """Smoothens an image"""
    smoothed = cv2.bilateralFilter(image, 9, 75, 75)
    return smoothed


def threshold(image):
    """Applies thresholding to an image"""
    thresholded = cv2.adaptiveThreshold(
        image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )
    return thresholded


def denoise(image):
    """Denoises an image"""
    denoised = cv2.fastNlMeansDenoising(image, None, 10, 7, 21)
    return denoised


def contrast(image):
    """Increases the contrast of an image"""
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    contrasted = clahe.apply(image)
    return contrasted
