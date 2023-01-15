from torch.utils.data import DataLoader, Dataset
from PIL import Image
from pytorch_lightning.core import LightningDataModule
from pathlib import Path
import torch
import logging
import numpy as np
import skin_lesion_cad.data.data_augmentation as rand_augment
import torchvision.transforms as transforms
import skin_lesion_cad.data.transforms as extended_transforms
from torch.utils.data import DataLoader, WeightedRandomSampler
import cv2
from torchvision.models.regnet import RegNet_X_800MF_Weights
regNet_transf = RegNet_X_800MF_Weights.IMAGENET1K_V2.transforms(crop_size=224)

logger = logging.getLogger(__name__)

PRETRAINING_LABELS = {'ack':0,'bcc':1,'bkl':2,'def':3,'mel':4,'nev':5,'scc':6,'vac':7}

def worker_init_fn(worker_id):
    """PyTorch Dataloader worker init function for setting different seed to each worker process
    Args:
            worker_id: the id of current worker process
    """
    np.random.seed(np.random.get_state()[1][0] + worker_id)


class MelanomaDataset(Dataset):
    def __init__(self, base_dir=None, split='train', chall="chall2", num=None, cfg=None):
        if chall == 'pretraining':
            self._base_dir = base_dir/Path('chall1')
        else:
            self._base_dir = base_dir/Path(chall)
        self.sample_list = []
        self.split = split
        self.transform = transforms
        self.chall = chall
        self.cfg = cfg
        self.input_size = cfg.INPUT_SIZE
        self.val_sample_repeat_num = 0
        self.color_space = cfg.COLOR_SPACE
        if self.split == 'train':
            self.sample_list = list(
                (self._base_dir/Path("train")).rglob("*.jpg"))

        elif self.split == 'val':
            self.sample_list = list(
                (self._base_dir/Path("val")).rglob("*.jpg"))

        elif self.split == 'predict':
            self.sample_list = list(
                (self._base_dir/Path("testX")).rglob("*.jpg"))

        if num is not None and self.split == "train":
            self.sample_list = self.sample_list[:num]
        print("total {} samples".format(len(self.sample_list)))

    def __len__(self):
        return len(self.sample_list)
    
    def get_sample_weights(self):
        if self.chall=="chall2":
            weight_map = {0:self.cfg.SAMPLING_WEIGHT[0], 1:self.cfg.SAMPLING_WEIGHT[1], 2:self.cfg.SAMPLING_WEIGHT[2]}
        elif self.chall=="chall1":
            weight_map = {0:self.cfg.SAMPLING_WEIGHT[0], 1:self.cfg.SAMPLING_WEIGHT[1]}
        else:
            raise Exception(
                "Argument chall must be either `chall1` or `chall2`")
        labels = [self.get_class(str(case.parent.stem)) for case in self.sample_list]
        return [*map(weight_map.get, labels)]

    def get_class(self, label):
        if self.chall == "chall2":
            if label == "bcc":
                return 0
            elif label == "mel":
                return 1
            elif label == "scc":
                return 2
            elif label == 'testX':
                return 0
            else:
                raise ValueError("class needs to be bcc, mel or scc")

        elif self.chall == "chall1":
            if label == "nevus":
                return 0
            else:
                return 1
        elif self.chall == "pretraining":
            return PRETRAINING_LABELS[label]
        else:
            raise Exception(
                "Argument chall must be either `chall1` or `chall2`")

    def __getitem__(self, idx):
        case = self.sample_list[idx]
        image = self._get_image(case)
        
        if self.chall == "pretraining":
            img_label = case.stem[:3]
        else:
            img_label = str(case.parent.stem)
        
        label = self.get_class(img_label)
        
        image = self.image_transform(image, idx)
        if self.split != "predict":
            sample = {'image': image, 'label': torch.tensor(label),
                      'name': case.stem}
        else:
            sample = {'image': image,
                      'name': case.stem}

        sample["idx"] = idx
        return sample

    def image_transform(self, img, index):
        img = self.image_pre_process(img)

        if self.split == "train":
            img = self._train_transform(img, index)
        else:
            img = self._val_transform(img, index)
        img = self.image_post_process(img)
        return img

    def image_post_process(self, img):
        # change the format of 'img' to tensor, and change the storage order from 'H x W x C' to 'C x H x W'
        # change the value range of 'img' from [0, 255] to [0.0, 1.0]
        to_tensor = transforms.ToTensor()
        img = to_tensor(img)       
        if self.cfg.SAMPLER.FIX_MEAN_VAR.ENABLE:
            normalize = transforms.Normalize(torch.from_numpy(np.array(self.cfg.SAMPLER.FIX_MEAN_VAR.SET_MEAN)),
                                             torch.from_numpy(np.array(self.cfg.SAMPLER.FIX_MEAN_VAR.SET_VAR)))
        else:
            # return img#/255.0
            normalize = extended_transforms.NormalizePerImage()
        return normalize(img)

    def _train_transform(self, img, index):
        if self.cfg.SAMPLER.AUGMENT.NEED_AUGMENT:  # need data augmentation
            # need another image
            while True:
                rand = np.random.randint(0, len(self.sample_list))
                if rand != index:
                    break
            bg_name = self.sample_list[rand]
            img_bg = self._get_image(bg_name)

            img = self.data_augment_train(img, img_bg)
        else:
            img = self.data_transforms_train(img)

        return img

    def _val_transform(self, img, index):
        if self.val_sample_repeat_num == 0:     # simple center crop
            crop_method = transforms.CenterCrop(self.input_size)
            img = crop_method(img)
        else:
            idx = index % self.val_sample_repeat_num
            if self.cfg.SAMPLER.MULTI_CROP.ENABLE and idx < self.cfg.TRAIN.SAMPLER.MULTI_CROP.CROP_NUM:   # multi crop
                img = self._val_multi_crop(img, idx)
            else:               # multi scale
                if self.cfg.SAMPLER.MULTI_CROP.ENABLE:
                    idx -= self.cfg.SAMPLER.MULTI_CROP.CROP_NUM
                img = self._val_multi_scale(img, idx)
        # img.show()
        return img

    def _get_image(self, img_path):
        img = cv2.imread(str(img_path))
        if img is None:
            raise ValueError("Image {} is None".format(img_path))
        if self.color_space == "RGB":
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    def data_augment_train(self, img, img_bg):

        img = torch.from_numpy(np.array(img, dtype=np.uint8))
        img_bg = torch.from_numpy(np.array(img_bg, dtype=np.uint8))

        blank_replace = tuple(
            [i * 255.0 for i in self.cfg.SAMPLER.FIX_MEAN_VAR.SET_MEAN])

        if self.cfg.SAMPLER.AUGMENT.AUG_METHOD == 'rand':     # RandAugment
            img = rand_augment.distort_image_with_randaugment(img, self.cfg.SAMPLER.AUGMENT.AUG_METHOD,
                                                              self.cfg.SAMPLER.AUGMENT.AUG_LAYER_NUM,
                                                              blank_replace,
                                                              self.cfg.SAMPLER.AUGMENT.AUG_MAG)
        else:       # Modified RandAugments
            img = rand_augment.distort_image_with_modified_randaugment(img, self.cfg.SAMPLER.AUGMENT.AUG_METHOD,
                                                                       img_bg, blank_replace,
                                                                       self.cfg.SAMPLER.AUGMENT.AUG_MAG)

        # --- random crop ---
        img = Image.fromarray(img.numpy())
        # transforms.RandomCrop(self.input_size),
        crop_method = extended_transforms.RandomCropInRate(nsize=self.input_size,
                                                           rand_rate=(self.cfg.SAMPLER.MULTI_CROP.L_REGION,
                                                                      self.cfg.SAMPLER.MULTI_CROP.S_REGION))
        img = crop_method(img)

        # img.show()
        return img

    def data_transforms_train(self, img):
        # --- random crop ---
        # transforms.RandomCrop(self.input_size),
        crop_method = extended_transforms.RandomCropInRate(nsize=self.input_size,
                                                           rand_rate=(self.cfg.SAMPLER.MULTI_CROP.L_REGION,
                                                                      self.cfg.SAMPLER.MULTI_CROP.S_REGION))
        img = crop_method(img)

        rand_h_flip = transforms.RandomHorizontalFlip()
        rand_v_flip = transforms.RandomVerticalFlip()
        img = rand_h_flip(img)
        img = rand_v_flip(img)

        if not self.cfg.SAMPLER.COLOR_CONSTANCY:
            # Color distortion
            color_distort = transforms.ColorJitter(
                brightness=32. / 255., saturation=0.5)
            img = color_distort(img)
        # img.show()
        return img

    def image_pre_process(self, img):
        if self.cfg.SAMPLER.BORDER_CROP == "pixel":
            if self.cfg.SAMPLER.BORDER_CROP_PIXEL > 0:
                img = img[self.cfg.SAMPLER.BORDER_CROP_PIXEL:-self.cfg.SAMPLER.BORDER_CROP_PIXEL,
                          self.cfg.SAMPLER.BORDER_CROP_PIXEL:-self.cfg.SAMPLER.BORDER_CROP_PIXEL, :]
            img = Image.fromarray(img)
        else:
            img = Image.fromarray(img)
            if self.cfg.SAMPLER.BORDER_CROP_RATIO > 0.0:
                sz_0 = int(
                    img.size[0] * (1 - self.cfg.SAMPLER.BORDER_CROP_RATIO))
                sz_1 = int(
                    img.size[1] * (1 - self.cfg.SAMPLER.BORDER_CROP_RATIO))
                crop_method = transforms.CenterCrop((sz_0, sz_1))
                img = crop_method(img)

        if self.cfg.SAMPLER.IMAGE_RESIZE:
            # the short side of the input image resize to a fix size
            resizing = transforms.Resize(
                self.cfg.SAMPLER.IMAGE_RESIZE_SHORT)
            img = resizing(img)

        if self.cfg.SAMPLER.COLOR_CONSTANCY and self.cfg.SAMPLER.APPLY_COLOR_CONSTANCY:
            color_constancy = extended_transforms.ColorConstancy(
                power=self.cfg.SAMPLER.CONSTANCY_POWER,
                gamma=None if self.cfg.SAMPLER.CONSTANCY_GAMMA == 0.0 else self.cfg.SAMPLER.CONSTANCY_GAMMA
            )
            img = color_constancy(img)
        return img

    def _train_transform(self, img, index):
        if self.cfg.SAMPLER.AUGMENT.NEED_AUGMENT:  # need data augmentation
            # need another image
            while True:
                rand = np.random.randint(0, len(self.sample_list))
                if rand != index:
                    break
            bg_info = self.sample_list[rand]
            img_bg = self._get_image(bg_info)

            img = self.data_augment_train(img, img_bg)
        else:
            img = self.data_transforms_train(img)

        return img

    def _val_multi_crop(self, img, idx):
        img = torch.from_numpy(np.array(img, dtype=np.uint8))
        img_size = img.size()
        num = np.int32(np.sqrt(self.cfg.SAMPLER.MULTI_CROP.CROP_NUM))
        y_n = int(idx / num)
        x_n = idx % num
        if img_size[1] >= img_size[0]:
            x_region = int(
                img_size[1] * self.cfg.SAMPLER.MULTI_CROP.L_REGION)
            y_region = int(
                img_size[0] * self.cfg.SAMPLER.MULTI_CROP.S_REGION)
        else:
            x_region = int(
                img_size[1] * self.cfg.SAMPLER.MULTI_CROP.S_REGION)
            y_region = int(
                img_size[0] * self.cfg.SAMPLER.MULTI_CROP.L_REGION)
        if x_region < self.input_size[1]:
            x_region = self.input_size[1]
        if y_region < self.input_size[0]:
            y_region = self.input_size[0]
        x_cut = int((img_size[1] - x_region) / 2)
        y_cut = int((img_size[0] - y_region) / 2)

        x_loc = x_cut + int(x_n * (x_region - self.input_size[1]) / (num - 1))
        y_loc = y_cut + int(y_n * (y_region - self.input_size[0]) / (num - 1))
        # Then, apply current crop
        img = img[y_loc:y_loc + self.input_size[0],
                  x_loc:x_loc + self.input_size[1], :]
        img = Image.fromarray(img.numpy())
        return img

    def _val_multi_scale(self, img, idx):
        factor = float(
            self.cfg.SAMPLER.MULTI_SCALE.SCALE_NAME[idx][-3:]) / 100.0 + 1.0
        new_height = round(self.input_size[0] * factor)
        new_width = round(self.input_size[1] * factor)
        crop_method = transforms.CenterCrop((new_height, new_width))
        img = crop_method(img)
        img = img.resize(
            (self.input_size[1], self.input_size[0],), Image.ANTIALIAS)

        if "flip_x" in self.cfg.SAMPLER.MULTI_SCALE.SCALE_NAME[idx]:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
        elif "flip_y" in self.cfg.SAMPLER.MULTI_SCALE.SCALE_NAME[idx]:
            img = img.transpose(Image.FLIP_TOP_BOTTOM)
        elif "rotate_90" in self.cfg.SAMPLER.MULTI_SCALE.SCALE_NAME[idx]:
            img = img.transpose(Image.ROTATE_90)
        elif "rotate_270" in self.cfg.SAMPLER.MULTI_SCALE.SCALE_NAME[idx]:
            img = img.transpose(Image.ROTATE_270)
        return img


class MelanonaDatasetSimple(MelanomaDataset):
    def __init__(self, base_dir=None, split='train', chall="chall2", num=None, cfg=None):
        super().__init__(base_dir, split, chall, num, cfg)
        self.cfg = cfg

    def __getitem__(self, idx):
        case = self.sample_list[idx]
        image = self._get_image(case)
        label = self.get_class(str(case.parent.stem))
        image = self.image_transform(image, idx)
        if self.split != "predict":
            sample = {'image': image, 'label': torch.tensor(label),
                      'name': case.stem}
        else:
            sample = {'image': image,
                      'name': case.stem}

        sample["idx"] = idx

        return sample
    def image_transform(self, img, index):
        timage = torch.Tensor(img).permute(2, 1, 0).to(torch.uint8)
        torchimage = regNet_transf.forward(timage)
        return torchimage

class MelanomaDataModule(LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        
        if cfg.dataloader == 'simple':
            self.DataSet = MelanonaDatasetSimple
        else:
            self.DataSet = MelanomaDataset
            
        self.cfg = cfg
        if cfg.train and True:
            self.train_dataset = self.DataSet(
                cfg.data_dir, split="train", chall=cfg.chall, cfg=cfg)
            self.val_dataset = self.DataSet(
                cfg.data_dir, split="val", chall=cfg.chall, cfg=cfg)
            logger.info(
                f'len of train examples {len(self.train_dataset)}, len of val examples {len(self.val_dataset)}'
            )
        else:
            self.test_dataset = self.DataSet(
                cfg.data_dir, split="test", chall=cfg.chall, cfg=cfg)
            logger.info(f'len of test examples {len(self.test_dataset)}')

    def train_dataloader(self):
        if self.cfg.WEIGHTED_SAMPLING:
            weights = self.train_dataset.get_sample_weights()
            sampler = WeightedRandomSampler(weights, len(self.train_dataset))
            train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.cfg.train_batch_size,
            sampler=sampler,
            num_workers=self.cfg.train_num_workers)

        else:
            train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.cfg.train_batch_size,
            shuffle=True,
            num_workers=self.cfg.train_num_workers)

        return train_loader

    def val_dataloader(self):
        if self.cfg.WEIGHTED_SAMPLING:
            weights = self.val_dataset.get_sample_weights()
            sampler = WeightedRandomSampler(weights, len(self.val_dataset))
            val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.cfg.val_batch_size,
            sampler=sampler,
            num_workers=self.cfg.val_num_workers)

        else:
            val_loader = DataLoader(
                self.val_dataset,
                batch_size=self.cfg.val_batch_size,
                shuffle=False,
                num_workers=self.cfg.val_num_workers)
        return val_loader

    def test_dataloader(self):
        test_loader = DataLoader(
            self.test_dataset,
            batch_size=self.cfg.test_batch_size,
            shuffle=False,
            num_workers=self.cfg.test_num_workers)
        return test_loader

