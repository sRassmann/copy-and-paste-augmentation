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

from torch.utils.data import Dataset


class PatchCreator:

    # tolerance to compensate for rounding in the bounding box representation
    xmin_tol = 1
    xmax_tol = 2
    ymin_tol = 1
    ymax_tol = 2

    def __init__(self, img_dir, output_dir, coco, save_patches=True):
        self.img_dir = img_dir
        self.output_dir = output_dir
        self.coco = coco
        self.average_sizes = {}
        self.cat_indices = {}
        for cat in coco.cats.values():
            os.makedirs(os.path.join(self.output_dir, cat["name"]), exist_ok=True)
            self.cat_indices[cat["name"]] = 0  # count number of objects for indexing
            self.average_sizes[cat["name"]] = (0, 0)  # average over sizes
        self.cat_ids = coco.getCatIds()
        self.save = save_patches

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
            if self.save:
                Image.fromarray(obj).save(
                    os.path.join(self.output_dir, cat, f"{cat}-{i}-{an['id']}.png")
                )
            self.average_sizes[cat] = (
                (self.average_sizes[cat][0] * (i - 1) + w) / i,
                (self.average_sizes[cat][1] * (i - 1) + h) / i,
            )


class CopyPasteGenerator:
    """Given a directory containing segmented objects this class generates randomly
    composed images.

    The class is thought to be able to Generate images on the fly in order to
    work as dataset class."""

    # ratio in which the aspect ratio of each object can be manipulated
    aspect_ratio_distort = 0.82

    # standard transformation for individual objs completely at random
    random_obj_augmentation = A.Compose(
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

    # standard transformation for individual objs for box simulation
    box_obj_augmentation = A.Compose(
        [
            A.Rotate(
                limit=10, border_mode=cv2.BORDER_CONSTANT, value=255, mask_value=0, p=1
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
        ],
        p=0.85,
    )

    def __init__(
        self,
        patch_sizes,
        obj_dir=os.path.join(constants.path_to_copy_and_paste, "objs"),
        background_dir=os.path.join(constants.path_to_copy_and_paste, "backgrounds"),
        obj_augmentation="default",
    ):
        """
        :param patch_sizes: Dict with key = cat_name and (av_width, av_height) as value
        :param obj_dir: dir containing individual object patches
        :param background_dir: dir containing backgrounds
        :param obj_augmentation: function (albumentations.Compose) performed for
        augmentation of individual objects.
        """
        np.random.seed(42)
        self.patch_sizes = patch_sizes

        self.cats = np.array(
            [os.path.basename(x) for x in glob.glob(os.path.join(obj_dir, "*"))]
        )
        self.objs = {}
        for cat in self.cats:
            self.objs[cat] = np.array(glob.glob(os.path.join(obj_dir, cat, "*png")))
        self.background_dir = background_dir
        self.background = None
        self.change_background("collection_box.tif")
        self.h, self.w = (self.background.shape[0], self.background.shape[1])
        if obj_augmentation != "default":
            self.random_obj_augmentation = (
                obj_augmentation
                if obj_augmentation
                else A.core.transforms_interface.NoOp()
            )

    def generate_collection_box(self, empty_p=0.1, cat=None):
        """
        simulates collection boxes

        :param empty_p: probability of an empty field
        :param cat: object category id to place in the boxes (random category if None)
        :return:
            img: composed image
            img_mask: mask to composed images with mask values encoding the instances
            cats: COCO category ids of the instances
        """
        img, img_mask = self.create_background()
        cat = cat if cat else np.random.choice(self.cats)
        cats = []
        grid_n_x = int(np.floor(img.shape[1] / self.patch_sizes[cat][0]).squeeze())
        grid_n_y = int(np.floor(img.shape[0] / self.patch_sizes[cat][1]).squeeze())
        av_w, av_h = self.patch_sizes[cat]
        size = max(self.patch_sizes[cat])

        jitter = 1 / 15  # std defining amount of deviation from grid

        k = 0
        for j in range(grid_n_x - 1):
            for i in range(grid_n_y - 1):
                if np.random.random(1) < empty_p:  # skip placement
                    continue
                k += 1
                # choose object
                obj = Image.open(np.random.choice(self.objs[cat]))
                cats.append(cat)

                # rescale object
                obj = self._rescale_obj(obj, (0.95 * size, 1 / 0.95 * size), True)

                # extract the mask from alpha channel of the image
                obj_mask = np.array(obj.getchannel("A"))
                obj = np.array(obj.convert("RGB"))
                obj_mask, obj = self._pad_images(obj_mask, obj)

                # data augmentation on object level
                t = self.box_obj_augmentation(image=obj, mask=obj_mask)
                obj = t["image"]
                obj_mask = t["mask"]
                obj_mask = (obj_mask > 0).astype(np.uint32)

                # add jitter to exact placement
                x, y = (j * av_w, i * av_h)
                x += np.random.normal(av_w / 2, av_h * jitter)
                y += np.random.normal(av_h / 2, av_w * jitter)
                x = int(x)
                y = int(y)
                if (
                    x + obj.shape[1] < img.shape[1]
                    and y + obj.shape[0] < img.shape[0]
                    and x > 0
                    and y > 0
                ):
                    self._place_obj_on_image(img, img_mask, obj, obj_mask, x, y, k)
        img = skimage.filters.gaussian(img, sigma=0.25, multichannel=True)
        return img, img_mask.astype(np.int32), cats

    def generate_random(self, n_objs=15, size=None, force=True):
        """place objects completely at random"""
        img, img_mask = self.create_background()
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
            t = self.random_obj_augmentation(image=obj, mask=obj_mask)
            obj = t["image"]
            obj_mask = t["mask"]
            obj_mask = (obj_mask > 0).astype(np.uint32)

            # place object on image
            x = np.random.randint(
                low=img.shape[1] / 20,
                high=img.shape[1] - obj.shape[1] - img.shape[1] / 20,
            )
            y = np.random.randint(
                low=img.shape[0] / 20,
                high=img.shape[0] - obj.shape[0] - img.shape[1] / 20,
            )
            self._place_obj_on_image(img, img_mask, obj, obj_mask, x, y, i)
        img = skimage.filters.gaussian(img, sigma=0.25, multichannel=True)
        return img, img_mask.astype(np.int32), cats

    def create_background(self):
        img = self.background.copy()
        rescale_factor = np.random.uniform(2.5, 4)
        img = cv2.resize(
            img,
            (
                int(img.shape[0] * rescale_factor * np.random.uniform(0.95, 1.06)),
                int(img.shape[1] * rescale_factor * np.random.uniform(0.95, 1.06)),
            ),
            interpolation=cv2.INTER_AREA,
        )
        img_mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.int32)
        return img, img_mask

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
                t = self.random_obj_augmentation(image=obj, mask=mask)
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


# class copy_paste_dataset(Dataset):
#     def __init__(self):
#         pass
#         # configure dir etc
#
#     def __len__(self):
#         c = 420  # some constant value
#         return c
#
#     def __getitem__(self, idx):
#         np.random.seed(idx)  # use idx for seed for reproducibility ?
#         sample = {"file_name": "",
#                   "height": 1,
#                   "width": 1,
#                   "image_id": idx,
#                   "annotations": {
#
#                       }
#
#                   }  # dict defined by detectron
#         return sample


def main():
    import lib

    # coco = COCO("../data/anno/all_anno.json")
    # p = PatchCreator("../data/imgs/", "../data/masked_objs/", coco)
    #
    # for image in coco.imgs.values():
    #     p(image)
    # print(p.average_sizes)
    av_sizes = {
        "Mesembryhmus purpuralis": (316.79715302491087, 192.3629893238434),
        "Smerinthus ocellata": (655.6478873239437, 379.9718309859156),
        "bug": (109.78313253012044, 224.90361445783134),
    }
    # g.visualize_augmentations(g.get_categories())
    c = lib.coco_handler.coco_dataset("../data/anno/all_anno.json")
    cat_dict = c.get_categories()

    g = CopyPasteGenerator(av_sizes)
    c = lib.coco_handler.coco_dataset()
    c.set_categories(cat_dict)
    cats = [
        "Smerinthus ocellata",
        "Smerinthus ocellata",
        "Smerinthus ocellata",
        "Smerinthus ocellata",
        "Smerinthus ocellata",
        "Smerinthus ocellata",
        "Smerinthus ocellata",
    ]
    for i in range(1, 2):
        img, mask, cats = g.generate_collection_box(0.1)
        fig, axs = plt.subplots(1, 2)
        axs[0].imshow(img)
        axs[1].imshow(mask)
        plt.show()
        io.imsave(f"../data/generated/{i}.png", (img * 255).astype(np.uint8))
        Image.fromarray(mask.astype(np.uint16)).save(f"../data/generated_masks/{i}.png")
        print(cats)
    img = io.imread(f"../data/generated_masks/{1}.png")
    c.add_annotations_from_instance_mask(img, f"{1}.png", cats)
    c.show_annotations(f"../data/generated/")


if __name__ == "__main__":
    main()
