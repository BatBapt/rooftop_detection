import os
import torch
import numpy as np
from torchvision.io import read_image
import configuration as cfg
from torch.utils.data import Dataset
from torchvision import tv_tensors
from torchvision.ops.boxes import masks_to_boxes
from torchvision.transforms.v2 import functional as F
import random

class CustomDataset(Dataset):
    """
    Custom dataset to load the data
    """
    def __init__(self, subset, mode="masks", mean_pts=7, transforms=None):
        """
        Constructor
        :args: subset, str: train, test or val
        :args: mode, str: masks or polygones (never really tried but seems to work)
        :args: mean_pts, int: mean number of point for each polygone to acces the data (see data_preprocessing.py file). Only used with polygone's mode
        :args: transforms: transformation to apply for image and mask
        """
        self.subset = subset
        self.mode = mode
        self.mean_pts = mean_pts
        self.transforms = transforms

        self.image_dir = os.path.join(cfg.BASE_PATH, self.subset, "images")

        if self.mode == "masks" or self.mode == "deeplab":
            self.masks_dir = os.path.join(cfg.BASE_PATH, self.subset, "masks")
            if not os.path.exists(self.masks_dir):
                print(f"Error, masks folder does not exists: {self.masks_dir}")
                exit()
        elif self.mode == "polygones":
            if self.mean_pts is not None:
                self.labels_polygones = os.path.join(cfg.BASE_PATH, self.subset, f"labels_resampled_{self.mean_pts}")
                if not os.path.exists(self.labels_polygones):
                    print(f"Error, labels polygones folder does not exists: {self.labels_polygones}")
                    exit()
            else:
                print(f"Mean points arguments, mean_pts, is set to None")
                exit()

        self.image_files = [f for f in os.listdir(self.image_dir)]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        sample_name = self.image_files[idx]

        img_name = os.path.join(self.image_dir, sample_name)
        img = read_image(img_name).float()
        img = img / 255.0

        height, width = img.shape[1], img.shape[2]

        target = {}

        if self.mode == "masks":
            mask_name = os.path.join(self.masks_dir, sample_name.replace(".jpg", ".png"))
            mask = read_image(mask_name)[0]
            mask = mask == 255

            obj_ids = torch.unique(mask)
            obj_ids = obj_ids[obj_ids != 0]  # 0 = background
            num_objs = len(obj_ids)
            if num_objs == 0:
                boxes = torch.zeros((0, 4), dtype=torch.float32)
                masks = torch.zeros((1, height, width), dtype=torch.uint8)
                labels = torch.zeros((0,), dtype=torch.int64)
                iscrowd = torch.zeros((0,), dtype=torch.int64)
                area = torch.zeros((0,), dtype=torch.float32)
            else:
                masks = (mask == obj_ids[:, None, None]).to(dtype=torch.uint8)
                boxes = masks_to_boxes(masks)
                labels = torch.ones((num_objs,), dtype=torch.int64)
                iscrowd = torch.zeros((num_objs,), dtype=torch.int64)
                area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

            target["masks"] = tv_tensors.Mask(masks)

        elif self.mode == "deeplab":
            mask_name = os.path.join(self.masks_dir, sample_name.replace(".jpg", ".png"))
            masks = read_image(mask_name)[0]

            if self.transforms is not None:
                img, masks = self.transforms(img, masks)

            return img, masks.long()

        elif self.mode == "polygones":
            file_label = os.path.join(self.labels_polygones, sample_name.replace(".jpg", ".txt"))
            with open(file_label, "r") as f:
                boxes = []
                labels = []
                areas = []
                iscrowd = []
                lines = f.readlines()
                if len(lines) != 0:
                    for line in lines:
                        parts = line.strip().split()
                        points = list(map(float, parts[1:]))
                        points_array = np.array(points).reshape(-1, 2)

                        points_array[:, 0] *= width
                        points_array[:, 1] *= height

                        x_min = np.min(points_array[:, 0])
                        y_min = np.min(points_array[:, 1])
                        x_max = np.max(points_array[:, 0])
                        y_max = np.max(points_array[:, 1])
                        boxe = [x_min, y_min, x_max, y_max]
                        boxes.append(boxe)
                        areas.append((boxe[3] - boxe[1]) * (boxe[2] * boxe[0]))
                        labels.append(1)  # it's 0 but Detection model automaticaly use 0 as the background
                        iscrowd.append(0)
                else:
                    boxes = torch.zeros((0, 4), dtype=torch.float32)
                    labels = torch.zeros((0,), dtype=torch.int64)
                    areas = torch.zeros((0,), dtype=torch.float32)
                    iscrowd = torch.zeros((0,), dtype=torch.int64)

            boxes = torch.tensor(boxes, dtype=torch.float32)
            labels = torch.tensor(labels, dtype=torch.int64)
            area = torch.tensor(areas, dtype=torch.float32)

        target["boxes"] = tv_tensors.BoundingBoxes(boxes, format="XYXY", canvas_size=F.get_size(img))
        target["labels"] = labels
        target["image_id"] = torch.tensor([idx])
        target["area"] = torch.tensor(area, dtype=torch.float32)
        target["iscrowd"] = torch.tensor(iscrowd, dtype=torch.int64)

        img = tv_tensors.Image(img)

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from torchvision.utils import draw_bounding_boxes, draw_segmentation_masks

    custom_ds = CustomDataset(subset="train", mode="masks", mean_pts=7)
    idx = 18
    img, target = custom_ds[idx]
    if img.dtype == torch.float32:
        img = (img * 255).to(torch.uint8)
    
    exec = "show_mask"

    if exec == "show_mask":
        masks = target["masks"].bool()
        overlay = draw_segmentation_masks(img, masks, alpha=0.5, colors="red")

        plt.figure(figsize=(12, 8))
        plt.imshow(overlay.permute(1, 2, 0))
        plt.axis("off")
        plt.show()

    elif exec == "show_bbox":
        boxes = target["boxes"]

        boxes = torch.tensor(boxes, dtype=torch.int64)
        overlay = draw_bounding_boxes(img, boxes, width=2, colors="red")

        plt.figure(figsize=(12, 8))
        plt.imshow(overlay.permute(1, 2, 0))  # Permute pour passer de CxHxW à HxWxC
        plt.axis("off")
        plt.show()


