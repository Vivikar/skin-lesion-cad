import pytorch_lightning as pl
import torch
import torchmetrics
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from skin_lesion_cad.training.models.regNet import RegNetY
from skin_lesion_cad.training.models.swin import SwinModel
from torch.optim.lr_scheduler import ReduceLROnPlateau, LinearLR

from skin_lesion_cad.utils.training_utils import get_loss

class ConvTransformerEnsemble(pl.LightningModule):
    def __init__(self,
                 num_classes:int,
                 regnet_chkp_path:str,
                 swin_chkp_path:str,
                 loss:str,
                 device=None,
                 freeze:bool=True,
                 lr:float=1e-4,
                 ):
        super().__init__()
        
        # save hyperparameters
        self.regnet_chkp_path = regnet_chkp_path
        self.swin_chkp_path = swin_chkp_path
        self.freeze_backbones = freeze
        self.loss = loss
        self.criterion = get_loss(loss, device=device, num_classes=num_classes)
        self.num_classes = num_classes
        self.learning_rate = lr
        
        # load pretrained models
        self.regnet = RegNetY.load_from_checkpoint(regnet_chkp_path)
        self.swin = SwinModel.load_from_checkpoint(swin_chkp_path)
        

        
        if self.freeze_backbones:
            self.regnet.freeze()
            self.swin.freeze()
        
        # create classifier and merge in it the feature encodings
        regnet_enc_dim = self.regnet.model.fc.in_features
        swin_enc_dim = self.swin.model.head.in_features
        in_features = regnet_enc_dim + swin_enc_dim

        
        # self.classifier = torch.nn.Sequential(torch.nn.Linear(in_features, 512),
        #                                       torch.nn.Softmax(),
        #                                       #torch.nn.Dropout(0.2),
        #                                       torch.nn.Linear(512, num_classes))
        self.classifier = torch.nn.Linear(in_features, num_classes)

        # remove last softmax output layer and replace with Identity
        self.regnet.model.fc = torch.nn.Identity()
        self.swin.model.head = torch.nn.Identity()

        # set up metrics to train
        if self.num_classes == 2:
            self.train_acc = torchmetrics.Accuracy(task='multiclass',
                                                num_classes=num_classes, top_k=1)
            self.valid_acc = torchmetrics.Accuracy(task='multiclass',
                                                num_classes=num_classes, top_k=1)
        elif self.num_classes == 3:
            self.train_kappa = torchmetrics.CohenKappa(num_classes=self.num_classes)
            self.valid_kappa = torchmetrics.CohenKappa(num_classes=self.num_classes)


        self.save_hyperparameters()


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        
        sch = ReduceLROnPlateau(optimizer, 'min',
                                factor=0.2, patience=8)
        
        #learning rate scheduler
        return {"optimizer": optimizer,
                "lr_scheduler": {"scheduler": sch,
                                 "monitor":"val_loss"}}


    def forward(self, x):
        regnet_enc = self.regnet(x) # 64x3
        swin_enc = self.swin(x) 
        x = torch.cat((regnet_enc, swin_enc), dim=1)
        x = self.classifier(x)
        return x
    
    
    def training_step(self, batch, batch_idx):
        x = batch['image']
        y = batch['label']  
        batch_size = len(y)
        y_hat = self.forward(x) # notice we change .model to .forward
        
        loss = self.criterion(y_hat, y)
        # need to manually average loss per batch
        if self.loss == 'focal':
            loss = loss.mean()

        self.log("train_loss", loss, on_step=False,
                 on_epoch=True, prog_bar=True,
                 logger=True, batch_size=batch_size)
        
        if self.num_classes == 2:
            self.train_acc(y_hat, y)
            self.log('train_acc', self.train_acc,
                    on_step=True,  on_epoch=True,
                    prog_bar=True, logger=True,
                    batch_size=batch_size)
        elif self.num_classes == 3:
            self.train_kappa(y_hat, y)
            self.log('train_kappa', self.train_kappa,
                    on_step=True,  on_epoch=True,
                    prog_bar=True, logger=True,
                    batch_size=batch_size)
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch['image']
        y = batch['label']
        batch_size = len(y)
        y_hat = self.forward(x) # notice we change .model to .forward
        
        loss = self.criterion(y_hat, y)
        # need to manually average loss per batch
        if self.loss == 'focal':
            loss = loss.mean()
        self.log("val_loss", loss, batch_size=batch_size)
        
        if self.num_classes == 2:
            self.valid_acc(y_hat, y)
            self.log('valid_acc', self.valid_acc,
                 on_step=True, on_epoch=True,
                 prog_bar=True, logger=True,
                 batch_size=batch_size)
        elif self.num_classes == 3:
            self.valid_kappa(y_hat, y)
            self.log('valid_kappa', self.valid_kappa,
                    on_step=True,  on_epoch=True,
                    prog_bar=True, logger=True,
                    batch_size=batch_size)
            
    def training_epoch_end(self, training_step_outputs):
        # do something with all training_step outputs
        # for out in training_step_outputs:
        if self.loss == 'mwn':
            self.criterion.reset_epoch(self.current_epoch)
    
    def predict_step(self, batch, batch_idx):
        x = batch['image']
        names = batch['name']
        batch_size = len(names)
        y_hat =  [i.argmax().item() for i in self.forward(x)]
        return names,y_hat