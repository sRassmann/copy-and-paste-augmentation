import os
from pycocotools.coco import COCO
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
from skimage import measure
from shapely.geometry import Polygon, MultiPolygon

DESCRIPTION = "Data set for segmenting insects"
DEFAULT_LICENSE = {
    "id": 1,
    "name": "Attribution-NonCommercial-ShareAlike License",
    "url": "http://creativecommons.org/licenses/by-nc-sa/2.0/",
}

__all__ = ["coco_dataset"]


class coco_dataset:
    """builds coco dataset iteratively to obtain coco annotation file"""

    info = {}
    licenses = []
    images = []
    categories = []
    annotations = []
    n_images = 0
    n_categories = 0
    n_annotations = 0

    def __init__(self, path_to_existing_coco=None, info=None, license=None):
        """
        creates coco_dataset instance based on existing coco annotation or a dummy file
         (if path_to_existing_coco = None)
        """
        if not info:
            self.info = self.create_coco_info(
                contrib=["Sebastian Rassmann"], version="v0.0.0", descr=DESCRIPTION
            )
        if not license:
            self.licenses = [DEFAULT_LICENSE]
        if path_to_existing_coco:
            ann = COCO(path_to_existing_coco)
            self.images = list(ann.imgs.values())
            self.categories = list(ann.cats.values())
            self.annotations = list(ann.anns.values())

    def add_annotation_from_mask(
        self,
        mask_path,
        image_name,
        category_name,
        super_category_name="",
        img_license=1,
        min_area=0,
    ):
        """
        creates COCO formatted instance annotation and add it to the coco file

        Objects are separated if (1) intensity values (greyscale) differ or (2) if they
        are not connected.
        """
        mask = io.imread(mask_path)
        image_id = len(self.images)
        self.images.append(
            self.create_coco_image(
                img_license, image_name, mask.shape[0], mask.shape[1], image_id
            )
        )
        d_cats = self.get_categories()
        if category_name not in d_cats:
            cat_id = len(d_cats) + 1
            self.categories.append(
                self.create_coco_category(category_name, cat_id, super_category_name))
        else:
            cat_id = self.categories[self.get_categories().index(category_name)]["id"]
        self.mask_to_coco(mask, cat_id, image_id, min_area)

    def mask_to_coco(self, mask, cat_id, image_id, min_area=0, black_background=False):
        mask = np.flip(np.rot90(mask, 1), axis=0)  # empirically proven necessary
        if not black_background:
            mask = np.invert(mask)

        for cont in measure.find_contours(mask, 0.5, fully_connected="low",
                                          positive_orientation="low"):
            poly = Polygon(cont)
            if poly.area > min_area:
                # poly = poly.simplify(0.2, preserve_topology=False)
                self.annotations.append(
                    self.create_coco_segmentation(
                        poly,
                        entry_id=len(self.annotations),
                        image_id=image_id,
                        category_id=cat_id,
                    )
                )

    def to_json(self, path=None):
        """dump obj to coco-compatible JSON String"""
        import json

        coco_dict = {
            "info": self.info,
            "categories": self.categories,
            "images": self.images,
            "annotations": self.annotations,
            "licenses": self.licenses,
        }
        jayson = json.dumps(coco_dict)
        if path:
            with open(path, "w+") as f:
                f.write(json.dumps(coco_dict, indent=4, sort_keys=False))
        return jayson

    def show_annotations(self, data_dir="../data/imgs/", cat_names="all"):
        """verification method, dumps itself to json and reloads using COCO"""
        tmp_json_path = "../output/tmp/tmp_coco_anno.json"
        self.to_json(tmp_json_path)
        coco = COCO(tmp_json_path)
        if cat_names == "all":
            cat_ids = coco.getCatIds()
        else:
            cat_ids = coco.getCatIds(catIds=cat_names)
        for i in coco.imgs:
            annIds = coco.getAnnIds(imgIds=i, catIds=cat_ids, iscrowd=None)
            anns = coco.loadAnns(annIds)
            img = io.imread(os.path.join(data_dir, coco.imgs[i]["file_name"]))
            plt.axis("off")
            plt.imshow(img)
            coco.showAnns(anns)
            plt.show()

    def get_categories(self):
        return [cat_dict["name"] for cat_dict in self.categories]

    ## --- Methods to handle COCO compatible entry formatting ---

    @staticmethod
    def create_coco_segmentation(poly, entry_id, image_id, category_id, is_crowd=0):
        """create coco segmentation from shapely.geometry.polygon.Polygon"""
        segs = []
        if isinstance(poly, Polygon):
            segs.append(coco_dataset.poly_to_flat_list(poly))
        elif isinstance(poly, MultiPolygon):
            segs = []
            for p in list(poly):
                segs.append(coco_dataset.poly_to_flat_list(p))
        return {
            "id": entry_id,
            "image_id": image_id,
            "category_id": category_id,
            "iscrowd": is_crowd,
            "area": poly.area,
            "bbox": [*coco_dataset.convert_bb_to_coco(*poly.bounds)],
            "segmentation": segs,
        }

    @staticmethod
    def poly_to_flat_list(poly):
        return np.array(poly.exterior.coords).ravel().tolist()

    @staticmethod
    def convert_bb_to_coco(minx, miny, maxx, maxy):
        """converts bb from shapely to coco format"""
        width = round(maxx - minx, 1)
        height = round(maxy - miny, 1)
        return round(minx), round(miny), width, height

    @staticmethod
    def create_coco_category(cat_name, cat_id, super_cat=""):
        return {"supercategory": super_cat, "id": cat_id, "name": cat_name}

    @staticmethod
    def create_coco_image(license_id, filename, height, width, image_id):
        return {
            "img_license": license_id,
            "file_name": filename,
            "height": height,
            "width": width,
            "id": image_id,
        }

    @staticmethod
    def create_coco_info(
        descr="desc",
        url="url",
        version="v0.0.0",
        year=None,
        contrib="cont",
        date_created=None,
    ):
        if not year or not date_created:
            from datetime import date

            year = year if year else date.today().strftime("%Y")
            date_created = (
                date_created
                if date_created
                else date.today().strftime("%Y/%m/%d-%H:%M:%S")
            )
        return {
            "description": descr,
            "url": url,
            "version": version,
            "year": year,
            "contributor": contrib,
            "date_created": date_created,
        }

def main():
    c = coco_dataset("../data/anno/butterfly_anno.json")
    for i in range(1, 6):
        c.add_annotation_from_mask(
            f"../data/masks/bug_{i}.tif",
            f"bug_{i}.tif",
            f"bug_proxy_{i}",
            "bug",
            min_area=150,
        )
    c.show_annotations()
    c.to_json("../data/anno/all_anno.json")


if __name__ == "__main__":
    main()
