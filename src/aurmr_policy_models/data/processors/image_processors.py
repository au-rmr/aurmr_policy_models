
from transformers import ViTImageProcessor

def get_image_processors_for_training(image_size):
    return {
        "cam_d405_rgb": ViTImageProcessor(
            image_size=image_size,
            do_resize=True,
            do_normalize=True,
            do_rescale=True,
        )
    }

def get_image_processors_for_eval(image_size):
    return {
        "cam_d405_rgb": ViTImageProcessor(
            image_size=image_size,
            do_resize=True,
            do_normalize=True,
            do_rescale=True,
        )
    }