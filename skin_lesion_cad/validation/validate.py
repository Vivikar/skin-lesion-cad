from pytorch_lightning import LightningDataModule, Trainer, LightningModule
from skin_lesion_cad.training.models.regNet import RegNetY
import pandas as pd
from pathlib import Path

def validate2csv(hydra_logpath: Path,
                 melanoma_data_module,
                 model_cls: LightningModule,
                 checkpoint:str):
    """Validated the model and saves the results to a csv file.

    Args:
        hydra_logpath (Path): Path to the model 
            used for validation (.ckpt)
        melanoma_data_module (LightningDataModule): instance of the data module
            with train and validation data
    """
    # save final metrics
    trainer = Trainer(accelerator='gpu', devices=[0])
    model = model_cls.load_from_checkpoint(checkpoint_path=hydra_logpath / "lightning_logs/version_1/checkpoints/{checkpoint}.ckpt}")
    res_val = trainer.validate(model=model, dataloaders=melanoma_data_module.val_dataloader())
    res_train  = trainer.validate(model=model, dataloaders=melanoma_data_module.train_dataloader())
    res_train = [{k.replace('val', 'train'):v for k,v in i.items()} for i in res_train]
    res_csv = pd.DataFrame(res_val + res_train)
    res_csv.to_csv(hydra_logpath / "results.csv")
    
if __name__ == '__main__':
    