import pyrootutils

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
from pytorch_lightning import LightningModule
import torchvision.transforms as T
import torch.nn.functional as F

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

    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(cfg.model)

    ckpt = torch.load(cfg.ckpt_path, map_location=torch.device('cpu'))

    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    scripted_model = model.to_torchscript(method="script")
    torch.jit.save(scripted_model, f"{cfg.paths.output_dir}/model.script.pt")
    log.info(f"Saved traced model to {cfg.paths.output_dir}/model.script.pt")

    log.info(f"Loaded Model: {model}")

    transform = T.Compose(
            [
                T.ToTensor(),
                T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

    def recognize_digit(image):
        if image is None:
            return None
        image = transform(image).unsqueeze(0)
        logits = model(image)
        preds = F.softmax(logits, dim=1).squeeze(0).tolist()
        return {str(i): preds[i] for i in range(10)}

    classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    def recognize_image(image):
        if image is None:
            return None
        image = transform(image).unsqueeze(0)
        output = model(image)
        print(output)
        _, predicted = torch.max(output, 1)
        print('Predicted: ', ' '.join(f'{classes[predicted[j]]:5s}'
                              for j in range(1)))


        # print(classes[predicted[0]])      
        # return "cat"
        return str(classes[predicted[0]])

    im = gr.Image(shape=(32, 32))

    demo = gr.Interface(
        fn=recognize_image,
        inputs=[im],
        # outputs=[gr.Label(num_top_classes=10)],
        outputs=[gr.Label(str)],
        live=True,
    )

    demo.launch()

@hydra.main(version_base="1.2", config_path=root / "configs", config_name="demo.yaml")
def main(cfg: DictConfig) -> None:
    demo(cfg)

if __name__ == "__main__":
    main()