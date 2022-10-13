from numpy import dtype
import pyrootutils
import torchvision.transforms as transforms
import numpy as np

root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git", "pyproject.toml"],
    pythonpath=True,
    dotenv=True,
)

from typing import List, Tuple

import torch
import hydra
import gradio as gr
from omegaconf import DictConfig

from src import utils

log = utils.get_pylogger(__name__)

def demo(cfg: DictConfig) -> Tuple[dict, dict]:
    """Demo function.
    Args:
        cfg (DictConfig): Configuration composed by Hydra.

    Returns:
        Tuple[dict, dict]: Dict with metrics and dict with all instantiated objects.
    """

    assert cfg.ckpt_path

    log.info("Running Demo")

    log.info(f"Instantiating scripted model <{cfg.ckpt_path}>")
    model = torch.jit.load(cfg.ckpt_path)

    log.info(f"Loaded Model: {model}")
    
    classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    def recognize_image(image):
        if image is None:
            return None

        print(image.shape)
        demo_array = np.moveaxis(image, -1, 0)
        print(demo_array.shape)
        np_arr = np.array(demo_array)
        img_tensor2 = torch.Tensor(np_arr).unsqueeze(0)
        preds = model.forward_jit(img_tensor2)
        print("preds ",preds)
        return str(classes[preds[0]])

    im = gr.Image( shape=(32, 32))

    demo = gr.Interface(
        fn=recognize_image,
        inputs=[im],
        outputs=[gr.Label(str)],
        live=True,
    )

    demo.launch()

@hydra.main(
    version_base="1.2", config_path=root / "configs", config_name="demo_scripted.yaml"
)
def main(cfg: DictConfig) -> None:
    demo(cfg)

if __name__ == "__main__":
    main()