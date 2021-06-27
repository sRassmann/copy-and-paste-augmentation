"""
module to handle detectron2 integration of CAP (requires detectron2 installation)
"""
from google.colab.patches import cv2_imshow
from detectron2.config import configurable
from detectron2.engine.defaults import DefaultTrainer
from detectron2.data.build import build_detection_train_loader
from detectron2.data import detection_utils
from detectron2.config import CfgNode as CN
from detectron2.structures.boxes import BoxMode
from detectron2.evaluation import COCOEvaluator
from detectron2.engine.hooks import HookBase
from detectron2.utils.logger import log_every_n_seconds
from detectron2.data import DatasetMapper, build_detection_test_loader
from detectron2.utils import comm

from torch.utils.data import Dataset
from lib import copy_and_paste_augm, constants
import os
import torch
import numpy as np
import yaml
import pickle
import time
import datetime
import logging

from lib.copy_and_paste_augm import *


def add_cap_config(cfg: CN):
    """
    adds CAP options to cfg
    """
    cfg.CAP = CN()

    c = cfg.CAP

    c.PATCH_POOL = "data/copy_and_paste/objs/"
    c.BACKGROUND_DIR = "data/copy_and_paste/backgrounds/"
    c.BACKGROUND_ANNO = "data/copy_and_paste/backgrounds/background_anno.json"
    c.MAX_RESOLUTION = (1800, 1500)
    c.POOLS_CFG = "configs/insects_cap_pool_config.yaml"

    c.RANDOM_GENERATOR = CN()
    R = c.RANDOM_GENERATOR
    R.POOL_NAME = "random_pool"
    R.P = 0.4
    R.SKIP_IF_OVERLAP_RANGE = (0.2, 0.4)
    R.MAX_N_OBJS_PER_IMAGE = 150
    R.ASSUMED_OBJ_SIZE = 40000

    c.COLLECTION_BOX_GENERATOR = CN()
    B = c.COLLECTION_BOX_GENERATOR
    B.POOL_NAME = "collection_box_pool"
    B.P = 0.6
    B.SKIP_IF_OVERLAP_RANGE = (0.1, 0.2)
    B.MAX_N_OBJS_PER_IMAGE = 150
    B.GRID_JITTER = (0.1, 0.35)
    B.SPACE_JITTER = (0.6, 1.3)

    c.HIGH_QUALITY_GENERATOR = CN()
    H = c.HIGH_QUALITY_GENERATOR
    H.POOL_NAME = "hq_pool"
    H.P = 0.0
    H.SKIP_IF_OVERLAP_RANGE = (0.0, 0.1)
    H.MAX_N_OBJS_PER_IMAGE = 150

    c.PICKLE_PATH = None  # use serialized pickle obj if available


def create_output(cfg):
    """
    creates detectron compatible output dir from cfg
    """
    dirname = cfg.OUTPUT_DIR
    os.makedirs(dirname, exist_ok=True)
    open(os.path.join(dirname, "metrics.json"), "a").close()


class CapDataset(Dataset):
    """
    class representing a detectron2 compatible dataset wrapping configuration and
     calling of the individual CAP generators
    """

    @configurable
    def __init__(
        self,
        patch_pool: str,
        background_anno: str,
        background_dir: str,
        max_resolution: (int, int),
        pools_cfg: str,
        random_cap_cfg: CN,
        collection_box_cfg: CN,
        hq_cfg: CN,
        seed=0,
        length=15000,
    ):
        self.parent_seed = seed
        np.random.seed(seed)
        self.length = length
        self.patch_pool = self.init_base_patch_pool(patch_pool)

        self.background_pool = BackgroundPool(
            background_dir=background_dir,
            background_anno=background_anno,
            max_resolution=max_resolution,
        )

        assert random_cap_cfg.P + collection_box_cfg.P + hq_cfg.P == 1
        self.collection_box_p = collection_box_cfg.P
        self.random_p = random_cap_cfg.P
        self.hq_p = hq_cfg.P

        if collection_box_cfg.P > 0:
            self.collection_box_cpg = CollectionBoxGenerator(
                self.patch_pool,
                self.background_pool,
                scale_augment_dict=self.get_pool_cfg(
                    pools_cfg, collection_box_cfg.POOL_NAME
                ),
                max_n_objs=collection_box_cfg.MAX_N_OBJS_PER_IMAGE,
                skip_if_overlap_range=collection_box_cfg.SKIP_IF_OVERLAP_RANGE,
                grid_pos_jitter=collection_box_cfg.GRID_JITTER,
                space_jitter=collection_box_cfg.SPACE_JITTER,
            )

        if random_cap_cfg.P > 0:
            self.random_cpg = RandomGenerator(
                self.patch_pool,
                self.background_pool,
                scale_augment_dict=self.get_pool_cfg(
                    pools_cfg, random_cap_cfg.POOL_NAME
                ),
                max_n_objs=random_cap_cfg.MAX_N_OBJS_PER_IMAGE,
                skip_if_overlap_range=random_cap_cfg.SKIP_IF_OVERLAP_RANGE,
                assumed_obj_size=random_cap_cfg.ASSUMED_OBJ_SIZE,
            )

        if hq_cfg.P > 0:
            raise NotImplementedError

    @classmethod
    def from_config(cls, cfg):
        return {
            "patch_pool": cfg.CAP.PATCH_POOL,
            "background_anno": cfg.CAP.BACKGROUND_ANNO,
            "background_dir": cfg.CAP.BACKGROUND_DIR,
            "max_resolution": cfg.CAP.MAX_RESOLUTION,
            "pools_cfg": cfg.CAP.POOLS_CFG,
            "random_cap_cfg": cfg.CAP.RANDOM_GENERATOR,
            "collection_box_cfg": cfg.CAP.COLLECTION_BOX_GENERATOR,
            "hq_cfg": cfg.CAP.HIGH_QUALITY_GENERATOR,
            "length": cfg.SOLVER.MAX_ITER,
        }

    @staticmethod
    def get_pool_cfg(cfg_path, pool_key):
        with open(cfg_path, "r") as y:
            d = yaml.load(y, Loader=yaml.FullLoader)
        return d[pool_key]

    @staticmethod
    def init_base_patch_pool(obj_dir):
        """
        initializes pool of raw object patches reused in for all created generators
        """
        # TODO annotate as yaml, include splitting instances for training and testing
        dirs = [os.path.basename(x) for x in glob.glob(os.path.join(obj_dir, "*"))]
        cat_ids = [s.split("-")[0] for s in dirs]
        cat_labels = [s.split("-")[1] for s in dirs]
        # create patch pool dict
        return {
            cat_label: PatchPool(
                os.path.join(obj_dir, f"{cat_id}-{cat_label}"),
                cat_id=cat_id,
                cat_label=cat_label,
                aug_transforms=None,
                n_augmentations=0,  # only create Pool
                scale=1,
            )
            for cat_id, cat_label in zip(cat_ids, cat_labels)
        }

    @staticmethod
    def try_restore_from_pickle(cfg):
        """
        load existing serialized obj from the path
        """
        # TODO print to logger
        if cfg.CAP.PICKLE_PATH and os.path.exists(cfg.CAP.PICKLE_PATH):
            self = pickle.load(open(cfg.CAP.PICKLE_PATH, "rb"))
            # TODO assert equality of params
            return self
        else:
            CapDataset(cfg)

    def dump(self, cfg, path=None):
        """
        dump created pool as pickle obj and save path to cfg

        Args:
            cfg:  config
            path: path to save the pickle obj. If None, the path from the cfg is used.
             Note, that the pickle objs can become large, so using a tmp dir might be
             better.
        """
        path = path if path else cfg.CAP.PICKLE_PATH
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)
        cfg.CAP.PICKLE_PATH = path

    def __getitem__(self, idx):
        """
        returns an object from one of the configured CAP Generators.

        The passed idx is used as seed for reproducibility and added to the defined seed
         in the constructor. Hence, starting from the defined seed a deterministic
         sequence of generated images is generated and returned.
        """
        np.random.seed(self.parent_seed + idx)

        r = np.random.rand()
        if 0 <= r < self.collection_box_p:
            gen = self.collection_box_cpg
        elif r < self.collection_box_p + self.random_p:
            gen = self.random_cpg
        else:
            gen = self.hq_cpg

        image, image_masks, bboxs, cats = gen()

        dataset_dict = {}
        dataset_dict["image"] = torch.as_tensor(
            image.transpose(2, 0, 1).astype("float32")
        )
        anns = [
            {
                "segmentation": mask.astype(bool),
                "category_id": cat,
                "bbox": bbox,  # not used but still nice to have
                "bbox_mode": BoxMode.XYWH_ABS,
            }
            for mask, bbox, cat in zip(image_masks, bboxs, cats)
        ]
        instances = detection_utils.annotations_to_instances(
            anns, image.shape[:2], "bitmask"
        )
        dataset_dict["instances"] = instances
        dataset_dict["height"] = image.shape[0]
        dataset_dict["width"] = image.shape[1]
        return dataset_dict

    def __len__(self):
        return self.length

    def visualize_pools(self):
        if self.random_p > 0:
            print("Pool for Random Images")
            self.random_cpg.visualize_pool()
        if self.collection_box_p > 0:
            print("Pool for Collection Box Images")
            self.random_cpg.visualize_pool()
        if self.hq_p > 0:
            print("Pool for High Quality Images")
            self.hq_cpg.visualize_pool()

    @staticmethod
    def get_projected_mask(dataset_dict):
        """
        utility function to project the created list of mask into a single mask
        """
        if not len(dataset_dict["instances"].gt_boxes):
            return 0
        t = dataset_dict["instances"].gt_masks.tensor  # tensor n_objs x H X W
        n = t.numpy().astype(np.uint32).sum(axis=0)
        assert np.unique(n).shape[0] <= 2
        return n

    @staticmethod
    def show_image_and_mask(dataset_dict):
        cv2_imshow(dataset_dict["image"].permute(1, 2, 0).numpy())
        if len(dataset_dict["instances"].gt_boxes):
            cv2_imshow(CapDataset.get_projected_mask(dataset_dict) * 255)


class Trainer(DefaultTrainer):
    """
    Trainer constructing and using a CapDataset from the config as training data
    """

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        return COCOEvaluator(dataset_name, output_dir=output_folder)

    @classmethod
    def build_train_loader(cls, cfg):
        dataset = CapDataset.try_restore_from_pickle(cfg)
        train_loader = build_detection_train_loader(
            dataset, total_batch_size=1, mapper=None
        )
        return train_loader

    def build_hooks(self):
        hooks = super().build_hooks()
        hooks.insert(
            -1,
            LossEvalHook(
                self.cfg.TEST.EVAL_PERIOD,
                self.model,
                build_detection_test_loader(
                    self.cfg, self.cfg.DATASETS.TEST[0], DatasetMapper(self.cfg, True)
                ),
            ),
        )
        return hooks


class LossEvalHook(HookBase):
    """
    track validation loss, taken from
     https://gist.github.com/ortegatron/c0dad15e49c2b74de8bb09a5615d9f6b
    """

    def __init__(self, eval_period, model, data_loader):
        self._model = model
        self._period = eval_period
        self._data_loader = data_loader

    def _do_loss_eval(self):
        # Copying inference_on_dataset from evaluator.py
        total = len(self._data_loader)
        num_warmup = min(5, total - 1)
        start_time = time.perf_counter()
        total_compute_time = 0
        losses = []
        for idx, inputs in enumerate(self._data_loader):
            if idx == num_warmup:
                start_time = time.perf_counter()
                total_compute_time = 0
            start_compute_time = time.perf_counter()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            total_compute_time += time.perf_counter() - start_compute_time
            iters_after_start = idx + 1 - num_warmup * int(idx >= num_warmup)
            seconds_per_img = total_compute_time / iters_after_start
            if idx >= num_warmup * 2 or seconds_per_img > 5:
                total_seconds_per_img = (
                    time.perf_counter() - start_time
                ) / iters_after_start
                eta = datetime.timedelta(
                    seconds=int(total_seconds_per_img * (total - idx - 1))
                )
                log_every_n_seconds(
                    logging.INFO,
                    "Loss on Validation  done {}/{}. {:.4f} s / img. ETA={}".format(
                        idx + 1, total, seconds_per_img, str(eta)
                    ),
                    n=5,
                )
            loss_batch = self._get_loss(inputs)
            losses.append(loss_batch)
        mean_loss = np.mean(losses)
        self.trainer.storage.put_scalar("validation_loss", mean_loss)
        comm.synchronize()
        return losses

    def _get_loss(self, data):
        # How loss is calculated on train_loop
        metrics_dict = self._model(data)
        metrics_dict = {
            k: v.detach().cpu().item() if isinstance(v, torch.Tensor) else float(v)
            for k, v in metrics_dict.items()
        }
        total_losses_reduced = sum(loss for loss in metrics_dict.values())
        return total_losses_reduced

    def after_step(self):
        next_iter = self.trainer.iter + 1
        is_final = next_iter == self.trainer.max_iter
        if is_final or (self._period > 0 and next_iter % self._period == 0):
            self._do_loss_eval()
        self.trainer.storage.put_scalars(timetest=12)
