"""
class offering simple copy and paste data augmentation
 (compare https://arxiv.org/abs/2012.07177).
Optimized to integrate with detectron2 
"""
from abc import abstractmethod

import os
import numpy as np
from skimage import morphology
from tqdm import tqdm
from PIL import Image
import json

import lib.constants as constants

import glob
import cv2

import albumentations as A
import matplotlib.pyplot as plt

from torch.utils.data import Dataset


class PatchCreator:

    """class handling cutting out objects as image patches from the image bases the
    corresponding COCO instance annotation file"""

    # TODO refactor to opencv
    # tolerance to compensate for rounding in the bounding box representation
    xmin_tol = 1
    xmax_tol = 1
    ymin_tol = 1
    ymax_tol = 1

    def __init__(
        self,
        coco,
        img_dir=constants.path_to_imgs_dir,
        output_dir=os.path.join(constants.path_to_output_dir, "patches"),
        save_patches=True,
    ):
        """
        create Callable PatchCreator

        Args:
            coco (pycocotools.coco.COCO): loaded coco istance annotation
            img_dir: base dir of raw images
            output_dir: dst dir for output
            save_patches: Save patches
        """
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

    def __call__(self, coco_image, dilation=None) -> None:
        """
        create patches from coco image

        Args:
            coco_image: coco.imgs entry
            dilation: amount of morphological mask dilation
        """
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
            x2 = min(xmin + w + self.xmax_tol + 1, img.shape[1] - 1)
            y1 = max(ymin - self.ymin_tol, 0)
            y2 = min(ymin + h + self.ymax_tol + 1, img.shape[0] - 1)
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


class CopyPasteGenerator(Dataset):
    """
    Given a directory containing segmented objects (Patches) this class generates
    randomly composed images.

    The class is thought to be able to generate images on-the-fly in order to
    work as dataset class.
    """

    # Perform no data augmentation as default
    AUGMENT = A.Compose(
        [A.NoOp()],
        p=0.0,
    )

    def __init__(
        self,
        obj_dir,
        background_dir,
        background_anno="background_anno.json",
        output_resolution=(1800, 1500),
        max_n_objs=150,
        n_augmentations=0,
        scale=1,
    ):
        """
        Initialize abstract CopyPasteGenerator.

        This loads all available background images into RAM and initializes the
        placement frames.
        The objects are loaded, rescaled, and, if specified, augmentations are applied
        to the individual objs and stored in RAM.

        Args:
            obj_dir: path to base directory containing on directory for each obj category
            background_dir: path to directory containing usable background images
            background_anno: path to background image frame annotation json
            output_resolution (int, int): target resolution of the images (if None the
             original background image resolutions are used)
            max_n_objs: maximum number of objects placed in one generate image
            n_augmentations: number of augmented version per object
            scale: rescale factor for each object
        """
        self.res = output_resolution
        self.background_dir = background_dir
        self.grid_rects, self.backgrounds = self.init_backgrounds(background_anno)
        self.cats = [os.path.basename(x) for x in glob.glob(os.path.join(obj_dir, "*"))]
        self.max_n_objs = max_n_objs
        self.patches = self.create_patch_pool(obj_dir, n_augmentations, scale)

    @classmethod
    def create_from_existing_pool(cls):
        """create Generator that uses the same raw image pool as another existing obj"""
        raise NotImplementedError

    def create_patch_pool(self, obj_dir, n_augmentations=0, scale=1):
        return {
            cat: CopyPasteGenerator.PatchPool(
                os.path.join(obj_dir, cat),
                cat_label=cat,
                aug_transforms=self.AUGMENT,
                n_augmentations=n_augmentations,
                scale=scale,
            )
            for cat in self.cats
        }  # get image from patch pool with self.patches[cat_name][idx]

    def add_scaled_version(self, cat, scale=2, n_augmentations=1):
        """adds down-scaled version of defined category to the patch pool"""
        org_pool = self.patches[cat]
        self.patches[f"{cat}-{scale}"] = self.PatchPool.create_from_existing_pool(
            org_pool,
            aug_transforms=self.__class__.AUGMENT,
            n_augmentations=n_augmentations,
            scale=scale,
        )

    def generate(self) -> (np.ndarray, [np.ndarray], [int], [str], np.ndarray):
        """generate image of randomly place objects"""
        max_n_objs = self.max_n_objs
        img, img_mask, rects = self.get_background(
            np.random.randint(0, len(self.backgrounds))
        )
        cats = []
        bboxs = []
        instance_mask = []

        for r in np.random.permutation(len(rects)):
            rect_masks, rect_bbox, rect_cat = self.place_in_rect(
                img, img_mask, rects[r], max_n_objs
            )
            max_n_objs -= len(rect_masks)
            cats += rect_cat
            instance_mask += rect_masks
            bboxs += rect_bbox
        return img, instance_mask, bboxs, cats, img_mask

    @abstractmethod
    def place_in_rect(self, image, image_mask, frame_rect, max_n_objs) -> None:
        """
        Atomic obj placement routine, override this function for specific patterns of
        object placement

        It should paste images (e.g. using
        :func:`copy_and_paste_augm.CopyPasteGenerator.paste_object`) to the background
        image and image_mask, and return the generated annotations.

        Args:
            image (np.ndarray): (partially filled) background image on which objs are
             placed
            image_mask (np.ndarray): (partially filled) mask of the image
            frame_rect ([int, int, int, int]): rectangle defining frame in which the
             objects can be placed (as [xmin, ymin, w, h])
            max_n_objs: maximum number of objs to be placed in this particular frame

        Returns:
            (full_size_masks, bounding_boxes, cats):
                full_size_masks (list(np.ndarray)): list of instance masks
                bounding_boxes (list([int, int, int, int])): bounding boxes of the
                instances
                cats (list(int)): category ids of the instances
        """
        pass

    def init_backgrounds(self, background_anno):
        """load background and frame annotation and rescale"""
        rects = json.load(open(os.path.join(self.background_dir, background_anno)))
        grid_rects = {
            k: v
            for k, v in rects.items()
            if os.path.exists(os.path.join(self.background_dir, k))
        }
        backgrounds = {}
        for k, v in grid_rects.items():
            img = cv2.imread(os.path.join(self.background_dir, k))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            assert img.shape[2] == 3
            if self.res:
                # rescale image to max res
                fmax = max(self.res) / max(img.shape[:2])
                fmin = min(self.res) / min(img.shape[:2])
                rescale_factor = min(fmin, fmax)
                img = cv2.resize(
                    img,
                    (
                        int(img.shape[1] * rescale_factor),
                        int(img.shape[0] * rescale_factor),
                    ),
                    interpolation=cv2.INTER_AREA,
                )
                assert max(*self.res) + 1 >= max(img.shape[:2])
                assert min(*self.res) + 1 >= min(img.shape[:2])
            backgrounds[k] = img
            # rescale grid rectangle
            for i, l in enumerate(grid_rects[k]):
                x, y, w, h = l
                xmax = (x + w) * rescale_factor
                ymax = (y + h) * rescale_factor
                x *= rescale_factor
                y *= rescale_factor
                w = xmax - x
                h = ymax - y
                grid_rects[k][i] = [round(x), round(y), round(w), round(h)]
        return grid_rects, backgrounds

    def get_background(self, index):
        key = list(self.backgrounds.keys())[index]
        image = self.backgrounds[key].copy()
        rects = self.grid_rects[key]
        mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
        return image, mask, rects

    def __len__(self) -> int:
        """dummy length (substitutes epoch length) for Dataset compatibility"""
        return self.length

    def __getitem__(
        self, seed
    ) -> (np.ndarray, [np.ndarray], [[int, int, int, int]], [int]):
        """
        Generate image with specified seed

        Args:
            seed (int): seed used for np.random.seed

        Returns:
            (img, instance_masks, bboxs, cats):
                img (np.ndarray): generated image (rgb)
                instance_masks (list(np.ndarray)): instance masks
                bboxs (list([int, int, int, int])): bounding boxes of generated
                image
                cats (list(int)): category ids of the instances
        """
        np.random.seed(seed)
        img, instance_masks, bboxs, cats, _ = self.generate()
        return img, instance_masks, bboxs, cats

    def __call__(self):
        img, image_masks, bboxs, cats, _ = self.generate()
        return img, image_masks, bboxs, cats

    def visualize_pool(self):
        """shows examples for each pool"""
        for cat, pool in self.patches.items():
            pool.visualize_augmentations(3, title=cat)

    def get_total_pool_size(self):
        return sum([p.__len__() for p in self.patches.values()])

    @staticmethod
    def paste_object(
        image,
        image_mask,
        obj,
        obj_mask,
        x_min,
        y_min,
        skip_if_overlap_func=None,
    ) -> (np.ndarray, [int, int, int, int]) or None:
        """
        Paste an object on a background image

        Args:
            image (np.ndarray): background image (8 bit rgb)
            image_mask (np.ndarray): background image mask (8 bit greyscale)
             (used for handling overlaps)
            obj (np.ndarray): object (8 bit rgb)
            obj_mask (np.ndarray): mask of the object (8 bit greyscale)
            x_min (int): minimum x coord (upper limit) of the object on the background
             image
            y_min (int): minimum y (left limit) coord of the object on the background
             image
            skip_if_overlap_func (func(obj_area, visible_obj_area) -> bool): if the
             func is defined (not None) and evaluates to True the obj is not placed

        Returns:
            inserted object mask (might be partially covered) and coordinates or
             None if the obj was not placed
        """
        # set up coords -> image[y_min : y_max, x_min : x_max] is manipulated
        assert image.dtype == np.uint8 and image.shape[2] == 3
        assert image_mask.dtype == np.uint8
        assert obj.dtype == np.uint8 and obj.shape[2] == 3
        assert obj_mask.dtype == np.uint8

        h, w = obj.shape[0], obj.shape[1]
        x_max = x_min + w
        y_max = y_min + h

        if x_min < 0 or y_min < 0 or x_max >= image.shape[1] or y_max >= image.shape[0]:
            return None

        # handle overlap --> only place the object where there is none yet
        org_area = np.sum(obj_mask > 0)
        obj_mask = cv2.bitwise_and(
            obj_mask, cv2.bitwise_not(image_mask[y_min:y_max, x_min:x_max])
        )

        # if skip_if_overlap_func is set and evaluates to True the obj is not pasted
        visible_obj_area = np.sum(obj_mask > 0)
        if skip_if_overlap_func and skip_if_overlap_func(org_area, visible_obj_area):
            print("skipped")
            return None

        # select background (unchanged area of the image)
        mask_inv = cv2.bitwise_not(obj_mask)
        bg = cv2.bitwise_and(
            image[y_min:y_max, x_min:x_max],
            image[y_min:y_max, x_min:x_max],
            mask=mask_inv,
        )

        # select foreground (newly placed object)
        fg = cv2.bitwise_and(obj, obj, mask=obj_mask)

        # place on obj and mask
        image[y_min:y_max, x_min:x_max] = cv2.add(bg, fg)
        image_mask[y_min:y_max, x_min:x_max] = cv2.bitwise_or(
            image_mask[y_min:y_max, x_min:x_max], obj_mask
        )

        # scale to full image size
        obj_mask_full_size = np.zeros(image.shape[:2])
        obj_mask_full_size[y_min:y_max, x_min:x_max] = obj_mask

        coords = [x_min, y_min, w, h]
        return obj_mask_full_size, coords

    class PatchPool:
        """represent the pool of cached patches for on class"""

        # TODO implement refreshing instances in separate thread

        def __init__(
            self,
            obj_dir,
            cat_label=None,
            aug_transforms=None,
            n_augmentations=1,
            min_max_size=50,
            scale=1,
        ):
            """
            Create PatchPool object from obj_dir to handle RAM caching for the pool of
            patches.

            Args:
                obj_dir: parent dir of patches, all  png files are considered
                cat_label: label of the category, will be returned together with
                 the obj and mask
                aug_transforms: Albumentations image and mask transformation (if None
                 the images is left unchanged)
                n_augmentations: number of augmented versions per patch
                min_max_size: min size of the larger patch side
                scale: scale factor by which the patch is resized
            """
            self.cat_label = cat_label
            self.augment = aug_transforms
            self.n_augment = n_augmentations
            if n_augmentations < 1:
                self.augment = None
                self.n_augment = 1
            self.replace_prob = 0.0
            self.scale = scale

            self.objs = []
            self.masks = []

            self.mean_h = 0  # real height (after augmentations)
            self.mean_w = 0  # real width (after augmentations)
            self.mean_max_size = 0  # mean size of largest side

            self.files = glob.glob(os.path.join(obj_dir, "*png"))
            self.org_image_pool = [
                self.open_image(file, min_max_size) for file in self.files
            ]
            [self.augment_images(img) for img in self.org_image_pool]
            self.init_image_stats()

        @classmethod
        def create_from_existing_pool(
            cls, parent, aug_transforms=None, n_augmentations=-1, scale=0.5
        ):
            """create pool from existing raw pool (e.g. for images on another - lower -
            scale)"""
            self = cls.__new__(cls)  # does not call __init__

            self.cat_label = parent.cat_label
            self.augment = aug_transforms
            self.n_augment = n_augmentations
            if n_augmentations < 1:
                self.augment = None
                self.n_augment = 1
            self.replace_prob = parent.replace_prob
            self.scale = scale

            self.objs = []
            self.masks = []

            self.mean_h = 0  # real height (after augmentations)
            self.mean_w = 0  # real width (after augmentations)
            self.mean_max_size = 0  # mean size of largest side

            self.org_image_pool = parent.org_image_pool  # use same original pool
            self.files = parent.files.copy()  # currently unused, required for reloading
            [self.augment_images(img) for img in self.org_image_pool]
            self.init_image_stats()
            return self

        def augment_images(self, obj_raw) -> None:
            """performs self.n_augment random augmentations, appends the resulting
            images to the pool, and crops images and masks to the object bounding
            box"""
            obj_raw, mask_raw = self.split_image_and_mask(obj_raw)

            if self.augment:
                h, w = obj_raw.shape[:2]
                l = max(h, w)
                ymin, xmin = (l - h // 2, l - w // 2)
                ymax, xmax = (ymin + h, xmin + w)

                # pad images
                obj_pad = np.zeros((l * 2, l * 2, 3), dtype=np.uint8)
                obj_pad[ymin:ymax, xmin:xmax, :] = obj_raw
                mask_pad = np.zeros((l * 2, l * 2), dtype=np.uint8)
                mask_pad[ymin:ymax, xmin:xmax] = mask_raw

                # augment
                for _ in range(self.n_augment):
                    t = self.augment(image=obj_pad, mask=mask_pad)
                    obj = t["image"]
                    mask = t["mask"]
                    assert obj.dtype == np.uint8
                    assert mask.dtype == np.uint8
                    self.compress_and_append(obj, mask)
            else:
                self.compress_and_append(obj_raw, mask_raw)

        def compress_and_append(self, img, mask) -> None:
            """rescale image and mask and crop to the bounding box of the mask"""
            if self.scale != 1:  # rescale obj and mask
                img = cv2.resize(
                    img,
                    (img.shape[1] * self.scale, img.shape[0] * self.scale),
                    interpolation=cv2.INTER_AREA,
                )
                mask = cv2.resize(
                    mask,
                    (mask.shape[1] * self.scale, mask.shape[0] * self.scale),
                    interpolation=cv2.INTER_AREA,
                )

            # crop to bounding box of augmented mask
            img, mask = self.crop_to_bb(img, mask)
            self.objs.append(img)
            self.masks.append(mask)

        def init_image_stats(self) -> None:
            shapes = np.array([x.shape[:2] for x in self.org_image_pool])
            self.mean_h, self.mean_w = tuple(np.mean(shapes, axis=0))
            max_sizes = np.max(shapes)
            self.mean_max_size = np.mean(max_sizes)

        @staticmethod
        def open_image(file, min_max_size) -> np.ndarray:
            """open images, handles encoding and scales the larger size up to
            min_max_size"""
            img = cv2.imread(file, cv2.IMREAD_UNCHANGED)
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA)

            l = max(img.shape[:2])
            if l < min_max_size:
                rescale_factor = int(2 ** np.ceil(-np.log2(l / min_max_size)))
                img = cv2.resize(
                    img,
                    (img.shape[1] * rescale_factor, img.shape[0] * rescale_factor),
                    interpolation=cv2.INTER_AREA,
                )
            return img

        @staticmethod
        def crop_to_bb(obj, mask) -> (np.ndarray, np.ndarray):
            """crop the image and its mask to obj"""
            x = np.nonzero(np.max(mask, axis=0))
            xmin, xmax = (np.min(x), np.max(x) + 1)
            y = np.nonzero(np.max(mask, axis=1))
            ymin, ymax = (np.min(y), np.max(y) + 1)
            obj = obj[ymin:ymax, xmin:xmax, :]
            mask = mask[ymin:ymax, xmin:xmax]
            return obj, mask

        @staticmethod
        def split_image_and_mask(obj) -> (np.ndarray, np.ndarray):
            """split 4 ch (rgba) .png into image (rgb) and mask (a)"""
            rgb = obj[:, :, :3]
            a = obj[:, :, 3]
            return rgb, a

        def replace_image(self, idx):
            """replaces images at idx with new version (in new thread)"""
            raise NotImplementedError

        def to_coco_dataset(self):
            """generate image and convert to COCO annotation file"""
            raise NotImplementedError

        def __getitem__(self, idx) -> (np.ndarray, np.ndarray):
            """if idx out of bounds a random image is returned, img is returned as rgb"""
            if not 0 <= idx < self.__len__():
                idx = np.random.randint(0, self.__len__())
            if np.random.rand() < self.replace_prob:
                self.replace_image(idx)
            return self.objs[idx], self.masks[idx], self.cat_label

        def __len__(self) -> int:
            return len(self.objs)

        def get_mean_height(self):
            return self.mean_h

        def get_mean_width(self):
            return self.mean_w

        def visualize_augmentations(self, n_examples=3, title=None):
            """show example of augmentations from current image pool"""
            # TODO avoid scaling images by pasting into square frame etc.
            n_cols = self.n_augment if self.n_augment > 0 else 1
            fig, axs = plt.subplots(n_examples, n_cols, figsize=(13, 8))
            axs = axs.reshape(-1, self.n_augment)  # in case self.n_augment == 1
            for i in range(n_examples):
                n = np.random.randint(0, self.__len__() / self.n_augment)  # choose obj
                for j in range(self.n_augment):
                    k = n * self.n_augment + j
                    obj, mask, _ = self[k]
                    mask = cv2.bitwise_not(mask)
                    obj[mask != 0] = (255, 255, 255)
                    axs[i, j].axis("off")
                    axs[i, j].imshow(obj)
            if not title:
                title = self[k][2]  # use saved name as label
            fig.suptitle(title, fontsize=16)
            plt.show()


class RandomGenerator(CopyPasteGenerator):
    """
    Places the images randomly with the only restrictions in the case the
    skip_if_overlap function is specified.
    """

    # standard transformation for individual objs completely at random
    # TODO add rescale parameter here, if the image is rotated quality is lost anyways
    AUGMENT = A.Compose(
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
        ],
        p=0.85,
    )

    def __init__(
        self,
        obj_dir,
        background_dir,
        background_anno="background_anno.json",
        output_resolution=(1800, 1500),
        max_n_objs=150,
        n_augmentations=5,
        scale=1,
        assumed_obj_size=300 * 300,
    ):
        """
        Extends overwritten init methods with the following args

        Args:
            assumed_obj_size: heuristic parameter to specify object density
        """
        super().__init__(
            obj_dir,
            background_dir,
            background_anno=background_anno,
            output_resolution=output_resolution,
            max_n_objs=max_n_objs,
            n_augmentations=n_augmentations,
            scale=scale,
        )
        # [self.add_scaled_version(cat, scale=1/4, n_augmentations=5) for cat in self.cats]
        # [self.add_scaled_version(cat, scale=1/8, n_augmentations=5) for cat in self.cats]

        self.assumed_obj_size = assumed_obj_size
        self.skip_if_overlap_func = (
            lambda obj_a, vis_a: np.random.rand() < 4 * (obj_a - vis_a) / obj_a
        )

    def place_in_rect(self, image, image_mask, frame_rect, max_n_objs) -> None:
        """
        Implementation of abstract place_in_rect function in parent.
        """
        # create list of objs from different pools (-> cats)
        artificial_pool = []

        rect_x, rect_y, rect_w, rect_h = frame_rect
        n_obj = rect_h * rect_w // self.assumed_obj_size  # heuristic param for obj size

        for i in range(n_obj):
            cat = np.random.choice(list(self.patches.keys()))
            pool = self.patches[cat]
            artificial_pool.append(pool[np.random.randint(0, len(pool))])

        full_size_masks = []
        bounding_boxes = []
        cats = []

        for i, p in enumerate(artificial_pool):
            if i > max_n_objs:
                break
            obj, obj_mask, cat = p
            if rect_w - obj.shape[1] < 0 or rect_h - obj.shape[0] < 0:
                continue
            x = np.random.randint(rect_x, rect_x + rect_w - obj.shape[1])
            y = np.random.randint(rect_y, rect_y + rect_h - obj.shape[0])

            pasted = self.paste_object(
                image, image_mask, obj, obj_mask, x, y, self.skip_if_overlap_func
            )
            if pasted:
                full_size_mask, bb = pasted
                full_size_masks.append(full_size_mask)
                bounding_boxes.append(bb)
                cats.append(cat)
        return full_size_masks, bounding_boxes, cats


class CollectionBoxGenerator(CopyPasteGenerator):
    """
    Chooses a single category at a single scale and places objs from this pool in a
    grid layout within the frame.
    """

    # standard transformation for individual objs for box simulation
    AUGMENT = A.Compose(
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
        obj_dir,
        background_dir,
        background_anno="background_anno.json",
        output_resolution=(1800, 1500),
        max_n_objs=150,
        n_augmentations=5,
        scale=1,
        grid_pos_jitter=0.2,
    ):
        """
        Extends overwritten init methods with the following args

        Args:
            grid_pos_jitter: heuristic parameter to specify jitter within the
        """
        super().__init__(
            obj_dir,
            background_dir,
            background_anno=background_anno,
            output_resolution=output_resolution,
            max_n_objs=max_n_objs,
            n_augmentations=n_augmentations,
            scale=scale,
        )
        [
            self.add_scaled_version(cat, scale=2, n_augmentations=n_augmentations)
            for cat in self.cats
        ]
        self.skip_if_overlap_func = (
            lambda obj_a, vis_a: np.random.rand() < 6 * (obj_a - vis_a) / obj_a
        )
        self.skip_if_overlap_func = None
        self.jitter = grid_pos_jitter

    def place_in_rect(self, image, image_mask, frame_rect, max_n_objs) -> None:
        """
        Implementation of abstract place_in_rect function in parent.
        """
        # choose random cat and scale
        cat = np.random.choice(self.cats)
        keys = [key for key in self.patches.keys() if key.startswith(cat)]
        pool = self.patches[np.random.choice(keys)]

        rect_x, rect_y, rect_w, rect_h = frame_rect
        av_w, av_h = pool.get_mean_width(), pool.get_mean_height()
        grid_n_x = rect_w // int(av_w)
        grid_n_y = rect_h // int(av_h)

        full_size_masks = []
        bounding_boxes = []
        cats = []

        k = 0
        for j in range(grid_n_x - 1):
            for i in range(grid_n_y - 1):
                k += 1
                obj, obj_mask, cat = pool[-1]  # get random patch

                # calculate location of the center
                x = rect_x + j * av_w + av_w // 2
                y = rect_y + i * av_h + av_h // 2

                # calculate left / lower limit and add some jitter
                x -= np.random.normal(obj.shape[1] / 2, self.jitter)
                y -= np.random.normal(obj.shape[0] / 2, self.jitter)
                x, y = int(x), int(y)

                # check if the instance is still in the frame
                if (
                    x + obj.shape[1] > rect_x + rect_w
                    or y + obj.shape[0] > rect_y + rect_h
                    or x < rect_x
                    or y < rect_y
                ):
                    continue

                pasted = self.paste_object(
                    image, image_mask, obj, obj_mask, x, y, self.skip_if_overlap_func
                )
                if pasted:
                    full_size_mask, bb = pasted
                    full_size_masks.append(full_size_mask)
                    bounding_boxes.append(bb)
                    cats.append(cat)
        return full_size_masks, bounding_boxes, cats


def main():
    cpg = CollectionBoxGenerator(
        os.path.join(constants.path_to_copy_and_paste, "objs"),
        os.path.join(constants.path_to_copy_and_paste, "backgrounds"),
        n_augmentations=1,
        grid_pos_jitter=0.05,
    )
    cpg.add_scaled_version("Smerinthus ocellata", 2, 10)
    cpg.add_scaled_version("bug", 2, 3)
    print(cpg.get_total_pool_size())
    for _ in range(10):
        img, image_masks, bboxs, cats, image_mask = cpg.generate()
        plt.imshow(img)
        plt.show()
        plt.imshow(image_mask)
    cpg.visualize_pool()


if __name__ == "__main__":
    main()
