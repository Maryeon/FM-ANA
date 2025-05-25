import os
import os.path as osp
import json
import random
import math

import torch
import torch.nn.functional as F
import torchvision.transforms.v2 as T
from torchvision.transforms.v2.functional import InterpolationMode
from torchvision.utils import save_image
from PIL import Image, ImageFilter

from collections import defaultdict


class BaseDataset:
    ac_codes = [
        "AC-1", "AC-2", "AC-3", "AC-4,5",
        "AC-6", "AC-7", "AC-8,9,10", "AC-11,12", "AC-13,14", "AC-15",
        "AC-16", "AC-17", "AC-18,19,20", "AC-21", "AC-22",
        "AC-23", "AC-24", "AC-25", "AC-26", "AC-27",
        "AC-28", "AC-29"
    ]
    pattern_names = [
        "Homogeneous", "Dense fine speckled", "Centromere", "Speckled",
        "Multiple", "Few", "Nucleolar", "Nuclear envelope", "Pleomorphic",
        "Linear", "Filamentous", "Segmental", "Cytoplasmic speckled",
        "AMA", "Golgi", "Rods and rings", "Cemtrosome", "Spindle fibers",
        "NuMA", "Intercellular", "Mitotic chromosomal", "Topo I"
    ]
    def __init__(self):
        pass

    @property
    def num_labels(self):
        return len(self.ac_codes)
    
    def prediction_to_labels(self, pred):
        pred = pred.cpu().numpy()

        return [self.ac_codes[i] for i, p in enumerate(pred) if p]


class Anahep2LabelDataset(BaseDataset):
    def __init__(self, root_path, label_file, split):
        self.root_path = root_path
        self.split = split

        with open(label_file, "r") as f:
            labels = json.load(f)[split]

        self.data = []
        self.img_ids = []
        self.num_crop_imgs = 0
        for img_id in list(labels.keys()):
            image_label = self._convert_to_multi_hot_label(labels[img_id]["image_label"])
            crop_image_files = []
            crop_img_labels = []
            for crop_item in labels[img_id]["crops"]:
                crop_image_files.append(crop_item["id"])
                crop_img_labels.append(self._convert_to_multi_hot_label(crop_item["label"]))
                self.num_crop_imgs += 1
            crop_img_labels = torch.stack(crop_img_labels, dim=0)

            self.data.append({
                "img_id": img_id,
                "crop_image_files": crop_image_files,
                "crop_image_labels": crop_img_labels,
                "image_label": image_label
            })

    def __len__(self):
        return len(self.data)
    
    def _convert_to_multi_hot_label(self, ac_codes):
        label_idx = [self.ac_codes.index(l) for l in ac_codes]
        label_idx = torch.tensor(label_idx, dtype=torch.long)
        multi_hot_label = F.one_hot(label_idx, num_classes=len(self.ac_codes)).sum(dim=0).float()
        return multi_hot_label


class Anahep2FolderDataset(BaseDataset):
    def __init__(self, root_path):
        self.root_path = root_path

        self.data = []
        self.img_ids = []
        self.num_crop_imgs = 0
        for img_id in os.listdir(self.root_path):
            crop_image_files = os.listdir(osp.join(self.root_path, img_id))

            self.data.append({
                "img_id": img_id,
                "crop_image_files": crop_image_files
            })

    def __len__(self):
        return len(self.data)


class GaussianBlur(object):
    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x
    

class SequentialSampler(object):
    def __init__(self, indices):
        self.indices = indices
        random.shuffle(self.indices)

    def __len__(self):
        return len(self.indices)
    
    def __iter__(self):
        yield from self.indices

    
class Anahep2DatasetTrain(Anahep2LabelDataset):
    def __init__(self, root_path, label_file, image_size=224):
        super().__init__(root_path, label_file, "train")
        self.crop_data = []
        self.label_to_index = defaultdict(list)

        idx = 0
        for item in self.data:
            img_id = item["img_id"]
            for crop_image_file, crop_image_label in zip(item["crop_image_files"], item["crop_image_labels"]):
                self.crop_data.append((osp.join(img_id, crop_image_file), crop_image_label))
                for index_label in self._convert_to_index_labels(crop_image_label):
                    self.label_to_index[index_label] += [idx]
                idx += 1

        self.transform = T.Compose([
            T.RandomResizedCrop(size=image_size, scale=(0.5, 1.0), interpolation=InterpolationMode.BILINEAR, antialias=True),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomVerticalFlip(p=0.5),
            T.AutoAugment(),
            T.Resize(image_size),
            T.PILToTensor(),
            T.ToDtype(torch.float, scale=True),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            T.ToPureTensor()
        ])

    def _convert_to_index_labels(self, label):
        res = []
        for i in range(label.shape[0]):
            if label[i]:
                res.append(i)
        return res

    def __len__(self):
        return len(self.crop_data)

    def __getitem__(self, i):
        crop_image_file, label = self.crop_data[i]

        image = Image.open(osp.join(self.root_path, crop_image_file)).convert("RGB")
        image = self.transform(image)

        return image, label
    

class Anahep2DatasetEval(Anahep2LabelDataset):
    def __init__(self, root_path, label_file, split, image_size=224):
        super().__init__(root_path, label_file, split)
        self.transform = T.Compose([
            T.Resize(size=(image_size, image_size), interpolation=InterpolationMode.BILINEAR, antialias=True),
            T.PILToTensor(),
            T.ToDtype(torch.float, scale=True),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            T.ToPureTensor()
        ])

    def __getitem__(self, i):
        img_id = self.data[i]["img_id"]
        crop_image_files = self.data[i]["crop_image_files"]
        labels = self.data[i]["crop_image_labels"]

        images = [Image.open(osp.join(self.root_path, img_id, image_file)).convert("RGB") for image_file in crop_image_files]
        images = [self.transform(image) for image in images]
        images = torch.stack(images, dim=0)
        
        return images, labels, img_id
    

def eval_collate_fn(batch):
    images, labels, img_ids = list(zip(*batch))
    num_crops = torch.tensor([x.shape[0] for x in images], dtype=torch.long)
    images = torch.cat(images, dim=0)
    labels = torch.cat(labels, dim=0)

    return images, labels, num_crops, img_ids


class Anahep2DatasetPredict(Anahep2FolderDataset):
    def __init__(self, root_path, image_size=224):
        super().__init__(root_path)
        self.transform = T.Compose([
            T.Resize(size=(image_size, image_size), interpolation=InterpolationMode.BILINEAR, antialias=True),
            T.PILToTensor(),
            T.ToDtype(torch.float, scale=True),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            T.ToPureTensor()
        ])

    def __getitem__(self, i):
        img_id = self.data[i]["img_id"]
        crop_image_files = self.data[i]["crop_image_files"]

        images = [Image.open(osp.join(self.root_path, img_id, image_file)).convert("RGB") for image_file in crop_image_files]
        images = [self.transform(image) for image in images]
        images = torch.stack(images, dim=0)
        
        return images, img_id


def predict_collate_fn(batch):
    images, img_ids = list(zip(*batch))
    num_crops = torch.tensor([x.shape[0] for x in images], dtype=torch.long)
    images = torch.cat(images, dim=0)

    return images, img_ids, num_crops


if __name__ == "__main__":
    dataset = Anahep2DatasetTrain(
        "/path/to/crops",
        "/path/to/splits.json"
    )