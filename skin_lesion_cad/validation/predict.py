from pytorch_lightning import LightningDataModule, Trainer, LightningModule
from skin_lesion_cad.training.models.regNet import RegNetY
from skin_lesion_cad.training.models.swin import SwinModel
import pandas as pd
from pathlib import Path
import hydra
from omegaconf import DictConfig
from hydra.core.hydra_config import HydraConfig

root = Path("skin_lesion_cad").resolve()

@hydra.main(version_base="1.2", config_path=str(root/Path("config")), config_name="test_config.yaml")
def gen_test_results(cfg: DictConfig):

    hydra_logpath = Path(HydraConfig.get().run.dir).resolve()
    checkpoint_path = Path(cfg.checkpoint_path)
    melanoma_data_module = hydra.utils.instantiate(config=cfg.data)
    melanoma_data_module = melanoma_data_module
    melanoma_data_module.prepare_data()    

    model_cls = hydra.utils.instantiate(config=cfg.model)
    # model_cls = SwinModel(num_classes=3, model_class='swin_v2_t')
    trainer = Trainer(accelerator='gpu')

    model = model_cls.load_from_checkpoint(checkpoint_path=hydra_logpath /checkpoint_path)
    
    
    # res_val = trainer.predict(model=model, dataloaders=melanoma_data_module.val_dataloader())
    # img = [j for i in res_val for j in i[0]]
    # pred = [j for i in res_val for j in i[1]]
    # pd.DataFrame({"image":img,f"{HydraConfig.get().job.name}_pred":pred}).to_csv(hydra_logpath/"prediction_val.csv")
    # print(f"Prediction saved to: {hydra_logpath/'prediction_val.csv'}")

    
    
    res_test = trainer.predict(model=model, dataloaders=melanoma_data_module.test_dataloader())
    img = [j for i in res_test for j in i[0]]
    pred = [j for i in res_test for j in i[1]]
    pd.DataFrame({"image":img,f"{HydraConfig.get().job.name}_pred":pred}).to_csv(hydra_logpath/"prediction_test.csv")
    print(f"Prediction saved to: {hydra_logpath/'prediction_test.csv'}")

    
if __name__=="__main__":
    gen_test_results()