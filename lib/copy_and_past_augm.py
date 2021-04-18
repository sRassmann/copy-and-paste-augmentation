"""
class offering simple copy and paste data augmentation
 (compare https://arxiv.org/abs/2012.07177).
Currently optimized to work with COCO datasets.
"""

import skimage
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
                mask = morphology.dilation(
                    mask, np.ones((dilation, dilation))
                )
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


    def __init__(self, obj_dir, background_dir):
        # load annotation and set image dir
        pass

    def generate_random(self):
        """simulate overlaps"""
        pass

    def generate_aligned(self, sorted=False):
        """simulate boxes"""
        pass



def main():
    coco = COCO("../data/anno/all_anno.json")
    p = PatchCreator("../data/imgs/", "../data/masked_objs/", coco)

    image = coco.imgs[3]
    p(image)


if __name__ == "__main__":
    main()
