
from pathlib import Path
from typing import Optional

import hydra
import matplotlib.pyplot as plt
import numpy as np
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import LightningDataModule
from skin_lesion_cad.data.datasets import MelanomaDataset, MelanonaDatasetSimple
from skin_lesion_cad.data.transforms import DeNormalize
import cv2

from torchvision.models.regnet import RegNet_X_800MF_Weights
root = Path("skin_lesion_cad").resolve()


@hydra.main(version_base="1.2", config_path=root/Path("config"), config_name="train_config.yaml")
def main(cfg: DictConfig) -> Optional[float]:
    print(OmegaConf.to_yaml(cfg))
    # seed_everything(cfg.seed)

    # print(f"Instantiating datamodule <{cfg._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(
        config=cfg)
    data_module = datamodule.data
    data_module.prepare_data()
    for data in data_module.train_dataloader():
        
        image = data["image"]

        image = image.squeeze(0)

        # FOR VISUALIZATION USE train_batch_size: 1
        image = DeNormalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])(image)

        batch_size = 1
        image = image.numpy()
        # for batch sizes > 1
        if len(image.shape) == 4:
            batch_size = image.shape[0]
            image = image[0]
        
        image = image.transpose(2, 1, 0)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        if batch_size != 1:
            cv2.imshow(f"{data['name'][0]} {data['label'][0]}",
                    image)
        else:
            cv2.imshow(f"{data['name']} {data['label']}",
                    image)
        if cv2.waitKey(0) & 0xFF == ord('q'):
            break


if __name__ == "__main__":
    main()
