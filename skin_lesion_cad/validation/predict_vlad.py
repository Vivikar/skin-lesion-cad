from pytorch_lightning import LightningDataModule, Trainer, LightningModule
from skin_lesion_cad.training.models.regNet import RegNetY
from skin_lesion_cad.training.models.swin import SwinModel
import pandas as pd
from pathlib import Path
from skin_lesion_cad.data.datasets import MelanomaDataset, DataLoader
import hydra
from omegaconf import DictConfig
from hydra.core.hydra_config import HydraConfig
import omegaconf
from skin_lesion_cad.training.models.ensemble import ConvTransformerEnsemble

root = Path("skin_lesion_cad").resolve()

CHKP_PATH = '/home/user0/cad_vlanasi/skin-lesion-cad/outputs/ens_pretr_chall2_freeze_mwnl_pretext/lightning_logs/version_1/checkpoints/epoch=05-valid_kappa=0.9533.ckpt'
CHALLENGE = "chall2"
PRED_FOLDER = '/home/user0/cad_vlanasi/skin-lesion-cad/predictions'


def unravel_predictions(predictions):
    img = [j for i in predictions for j in i[0]]
    pred = [j for i in predictions for j in i[1]]
    res = pd.DataFrame({"image":img,
                        f"pred":pred})
    return res

def gen_test_results():
    cfg_path = Path(CHKP_PATH).parent.parent.parent.parent/'.hydra/config.yaml'
    cfg = omegaconf.OmegaConf.load(cfg_path)
    
    val_dataset = MelanomaDataset(base_dir=Path('/home/user0/cad_vlanasi/skin-lesion-cad/data/raw'),
                                   split='val',
                                   chall=CHALLENGE,
                                   num=None,
                                   cfg=cfg.data.cfg)
    val_loader = DataLoader(
            dataset=val_dataset,
            batch_size=64,
            shuffle=False,
            num_workers=10)
    
    test_dataset = MelanomaDataset(base_dir=Path('/home/user0/cad_vlanasi/skin-lesion-cad/data/raw'),
                                   split='predict',
                                   chall=CHALLENGE,
                                   num=None,
                                   cfg=cfg.data.cfg)
    test_loader = DataLoader(
            dataset=test_dataset,
            batch_size=64,
            shuffle=False,
            num_workers=10)
    
    model = ConvTransformerEnsemble.load_from_checkpoint(checkpoint_path=CHKP_PATH)
    
    trainer = Trainer(accelerator='gpu', devices=[0])
    
    
    test_predictions = unravel_predictions(trainer.predict(model, test_loader))
    valid_predictions =  unravel_predictions(trainer.predict(model, val_loader))
    
    
    name = Path(CHKP_PATH).parent.parent.parent.parent.name
    test_predictions.to_csv(f"{PRED_FOLDER}/{name}_test_predictions.csv")
    valid_predictions.to_csv(f"{PRED_FOLDER}/{name}_valid_predictions.csv")
    
    
if __name__=="__main__":
    gen_test_results()