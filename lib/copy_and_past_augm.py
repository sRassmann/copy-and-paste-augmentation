"""
class offering simple copy and paste data augmentation
 (compare https://arxiv.org/abs/2012.07177).
Currently optimized to work with COCO datasets.
"""

import skimage
from torchvision import transforms

import lib.coco_handler as ch
import os
from pycocotools.coco import COCO
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
from skimage import measure, morphology
from shapely.geometry import Polygon, MultiPolygon
from tqdm import tqdm
from PIL import Image

import lib.constants as constants

import glob
import cv2

import albumentations as A


class PatchCreator:

    # tolerance to compensate for rounding in the bounding box representation
    xmin_tol = 1
    xmax_tol = 2
    ymin_tol = 1
    ymax_tol = 2

    def __init__(self, img_dir, output_dir, coco):
        self.img_dir = img_dir
        self.output_dir = output_dir
        self.coco = coco
        self.cat_indices = {}
        for cat in coco.cats.values():
            os.makedirs(os.path.join(self.output_dir, cat["name"]), exist_ok=True)
            self.cat_indices[cat["name"]] = 0  # count number of objects for indexing
        self.cat_ids = coco.getCatIds()

    def __call__(self, coco_image, dilation=None):
        img = Image.open(os.path.join(self.img_dir, coco_image["file_name"]))
        img = img.convert("RGBA")
        img = np.asarray(img.convert("RGBA"))

        anns_ids = self.coco.getAnnIds(imgIds=coco_image["id"], iscrowd=None)
        anns = self.coco.loadAnns(anns_ids)

        for an in tqdm(anns):
            # crop bb from masks and images
            xmin, ymin, w, h = tuple([int(i) for i in an["bbox"]])
            # ensure bb bounds are not outside the image frame
            x1 = max(xmin - self.xmin_tol, 0)
            x2 = min(xmin + w + self.xmax_tol, img.shape[1] - 1)
            y1 = max(ymin - self.ymin_tol, 0)
            y2 = min(ymin + h + self.ymax_tol, img.shape[0] - 1)
            mask = self.coco.annToMask(an)[y1:y2, x1:x2]
            if dilation:
                mask = morphology.dilation(mask, np.ones((dilation, dilation)))
            obj = img[y1:y2, x1:x2].copy()

            # set unmasked corners to transparent
            obj[:, :, 3] = np.where(mask, obj[:, :, 3], 0)

            # save cropped image
            cat = self.coco.loadCats(ids=[an["category_id"]])[0]["name"]
            i = self.cat_indices[cat] + 1
            self.cat_indices[cat] = i
            Image.fromarray(obj).save(
                os.path.join(self.output_dir, cat, f"{cat}-{i}-{an['id']}.png")
            )


class CopyPasteGenerator:
    """Given a directory containing segmented objects this class generates randomly
    composed images.

    The class is thought to be able to Generate images on the fly in order to
    work as dataset class."""

    # ratio in which the aspect ratio of each object can be manipulated
    aspect_ratio_distort = 0.82

    # standard transformation for individual objs
    obj_augmentation = A.Compose(
        [
            A.Rotate(
                limit=360, border_mode=cv2.BORDER_CONSTANT, value=255, mask_value=0, p=1
            ),
            A.transforms.ColorJitter(
                brightness=0.3,
                contrast=0.1,
                saturation=0.1,
                hue=0.03,
                always_apply=False,
                p=0.5,
            ),
            A.transforms.FancyPCA(alpha=0.02, always_apply=False, p=0.5),
            A.augmentations.transforms.ToGray(p=0.1),
        ],
        p=0.85,
    )

    def __init__(
        self,
        obj_dir=os.path.join(constants.path_to_copy_and_paste, "objs"),
        background_dir=os.path.join(constants.path_to_copy_and_paste, "backgrounds"),
        obj_augmentation="default",
    ):
        np.random.seed(42)
        self.cats = np.array(
            [os.path.basename(x) for x in glob.glob(os.path.join(obj_dir, "*"))]
        )
        self.objs = {}
        for cat in self.cats:
            self.objs[cat] = np.array(glob.glob(os.path.join(obj_dir, cat, "*png")))
        self.background_dir = background_dir
        self.background = Image.new("L", (1200, 1200), 0)
        self.change_background("collection_box.tif")
        self.h, self.w = (self.background.shape[0], self.background.shape[1])
        if obj_augmentation != "default":
            self.obj_augmentation = (
                obj_augmentation
                if obj_augmentation
                else A.core.transforms_interface.NoOp()
            )

    def generate_aligned(self, background_dir):
        """simulate boxes"""
        raise NotImplementedError

    def generate_random(self, n_objs=15, size=None, force=True):
        """simulate overlaps"""
        img = self.background.copy()
        img_mask = np.zeros((img.shape[0], img.shape[1]))
        cats = []
        for i in range(1, n_objs + 1):
            # choose object
            cat = np.random.choice(self.cats)
            obj = Image.open(np.random.choice(self.objs[cat]))
            cats.append(cat)

            # rescale object
            if size:
                obj = self._rescale_obj(obj, size, force)

            # extract the mask from alpha channel of the image
            obj_mask = np.array(obj.getchannel("A"))
            obj = np.array(obj.convert("RGB"))
            obj_mask, obj = self._pad_images(obj_mask, obj)

            # data augmentation on object level
            t = self.obj_augmentation(image=obj, mask=obj_mask)
            obj = t["image"]
            obj_mask = t["mask"]
            obj_mask = (obj_mask > 0).astype(np.uint32)

            # place object on image
            x = np.random.randint(low=0, high=img.shape[1] - obj.shape[1])
            y = np.random.randint(low=0, high=img.shape[0] - obj.shape[0])
            self._place_obj_on_image(img, img_mask, obj, obj_mask, x, y, i)

        return img, img_mask, cats

    def visualize_augmentations(self, cats, size=(100, 100), n_examples=10):
        """show random augmentations for each specified cat"""
        cats = [cat for cat in cats if cat in self.cats]  # filter for existing cats
        fig, axs = plt.subplots(len(cats), n_examples, figsize=(13, 4))
        for i, cat in enumerate(cats):
            org_obj = Image.open(np.random.choice(self.objs[cat]))
            for j in range(n_examples):
                obj = self._rescale_obj(org_obj, size, True).copy()
                obj_mask = np.array(obj.getchannel("A"))
                obj = np.array(obj.convert("RGB"))
                mask, obj = self._pad_images(obj_mask, obj)
                t = self.obj_augmentation(image=obj, mask=mask)
                obj = t["image"]
                mask = (t["mask"] == 0).astype(np.uint32)[..., np.newaxis]
                obj = np.where(mask, 255, obj)
                axs[i, j].axis("off")
                axs[i, j].imshow(obj)
        plt.show()

    def change_background(self, name):
        """set new image (np.ndarray) as background"""
        self.background = io.imread(os.path.join(self.background_dir, name))
        self.h, self.w = (self.background.shape[0], self.background.shape[1])

    def get_categories(self):
        return list(self.cats)

    @staticmethod
    def _place_obj_on_image(img, img_mask, obj, obj_mask, x, y, mask_value):
        img_mask[y : y + obj.shape[0], x : x + obj.shape[1]] = np.where(
            obj_mask,
            obj_mask * mask_value,
            img_mask[y : y + obj.shape[0], x : x + obj.shape[1]],
        )
        obj_mask = obj_mask.reshape(*obj_mask.shape, 1)
        img[y : y + obj.shape[0], x : x + obj.shape[1], :] = (
            img[y : y + obj.shape[0], x : x + obj.shape[1], :] * (1 - obj_mask)
            + obj * obj_mask
        )

    @staticmethod
    def _pad_images(mask, obj):
        pad_len = int(np.ceil(np.sqrt(obj.shape[0] ** 2 + obj.shape[1] ** 2) / 4))
        mask = np.pad(
            mask,
            (
                (pad_len, pad_len),  # pad bottom - top
                (pad_len, pad_len),  # left - right
            ),
            mode="constant",
            constant_values=0,
        )
        obj = np.pad(
            obj,
            (
                (pad_len, pad_len),  # pad bottom - top
                (pad_len, pad_len),  # left - right
                (0, 0),  # channels
            ),
            mode="constant",
            constant_values=0,
        )
        return mask, obj

    def _rescale_obj(self, obj, size, force):
        """
        Randomly rescale single objects

        :param obj: PIL image crop to be manipulated
        :param size: range of desired size (uniform distribution)
        :param force: if False, objs are not up-scaled by more than 4 fold
        :return: rescaled object
        """
        obj_w, obj_h = (obj.size[0], obj.size[1])
        obj_size = max(obj_h, obj_w)
        target_size = np.random.uniform(size[0], size[1])
        scale = target_size / obj_size
        if not force:
            scale = min(scale, 4)
        obj = obj.resize(
            (
                round(
                    obj.size[0]
                    * scale
                    * np.random.uniform(
                        low=1 / self.aspect_ratio_distort,
                        high=self.aspect_ratio_distort,
                    )
                ),
                round(obj.size[1] * scale),
            )
        )
        return obj


def main():
    # coco = COCO("../data/anno/all_anno.json")
    # p = PatchCreator("../data/imgs/", "../data/masked_objs/", coco)
    #
    # image = coco.imgs[3]
    # p(image)
    g = CopyPasteGenerator()
    g.visualize_augmentations(g.get_categories())
    for _ in range(5):
        img, mask, cats = g.generate_random(size=(100, 220))
        fig, axs = plt.subplots(1, 2)
        axs[0].imshow(img)
        axs[1].imshow(mask)
        plt.show()


if __name__ == "__main__":
    main()
