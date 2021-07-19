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
from detectron2.data.transforms import ResizeShortestEdge
from detectron2.utils.visualizer import Visualizer

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
import glob
import cv2


def visualize_detectron2_loader(data_loader, cfg, n_examples=5, scale=1):
    """
    show images from train or test loader with COCO annotations based on
     https://github.com/facebookresearch/detectron2/blob/master/tools/visualize_data.py
    """
    for batch in data_loader:
        for per_image in batch:
            if n_examples == 0:
                return
            img = per_image["image"].permute(1, 2, 0).cpu().detach().numpy()
            img = detection_utils.convert_image_to_rgb(img, cfg.INPUT.FORMAT)
            visualizer = Visualizer(img, scale=scale)
            target_fields = per_image["instances"].get_fields()
            vis = visualizer.overlay_instances(
                boxes=target_fields.get("gt_boxes", None),
                masks=target_fields.get("gt_masks", None),
            )
            cv2_imshow(vis.get_image()[:, :, ::-1])
            cv2_imshow(
                cv2.resize(
                    img, (round(img.shape[1] * scale), round(img.shape[0] * scale))
                )
            )
            n_examples -= 1


def create_output(cfg):
    """
    creates detectron compatible output dir from cfg
    """
    dirname = cfg.OUTPUT_DIR
    os.makedirs(dirname, exist_ok=True)
    open(os.path.join(dirname, "metrics.json"), "a").close()


def get_bb(mask):
    """
    infer bounding box from mask (as detectron2.structures.boxes.BoxMode.XYXY_ABS)
    """
    x_proj = np.where(np.amax(mask, axis=0) > 0)
    xmin = np.min(x_proj)
    xmax = np.max(x_proj)

    y_proj = np.where(np.amax(mask, axis=1) > 0)
    ymin = np.min(y_proj)
    ymax = np.max(y_proj)
    return [xmin, ymin, xmax, ymax]


class build_dataset_dict:
    """
    build a dataset dictionary representing all images in a directory
    """

    def __init__(self, img_dir):
        self.img_dir = img_dir

    def __call__(self):
        assert os.listdir(self.img_dir)
        dicts = []
        for idx, f in enumerate(glob.glob(os.path.join(self.img_dir, "*.png"))):
            dicts.append(
                {
                    "file_name": f,
                    "image_id": idx,
                }
            )
        return dicts


class build_mapper:
    """
    build a mapper retrieving the mask from a RGBA image and applying the specified
     augmentations
    """

    def __init__(self, aug):
        self.aug = aug

    def __call__(self, dataset_dict):
        rgba = cv2.imread(dataset_dict["file_name"], cv2.IMREAD_UNCHANGED)
        img = rgba[:, :, :3]
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        mask = rgba[:, :, 3]

        augment = self.aug(image=img, mask=mask)
        img = augment["image"]
        mask = augment["mask"]

        dataset_dict["image"] = torch.as_tensor(
            img.transpose(2, 0, 1).astype("float32")
        )
        anns = [
            {
                "segmentation": mask.astype(bool),
                "category_id": 0,
                "bbox": get_bb(mask),
                "bbox_mode": BoxMode.XYXY_ABS,
            }
        ]
        instances = detection_utils.annotations_to_instances(
            anns, img.shape[:2], "bitmask"
        )
        dataset_dict["instances"] = instances
        dataset_dict["height"] = img.shape[0]
        dataset_dict["width"] = img.shape[1]
        return dataset_dict


class SingleInstanceRgbaDataset:
    def __init__(img_dir, aug):
        assert os.listdir(img_dir), "No images found in specified dir"
        self.img_dir = img_dir

    def get_rgba_dataset_dict(img_dir):
        assert os.listdir(img_dir)
        dicts = []
        for idx, f in enumerate(glob.glob(os.path.join(img_dir, "*.png"))):
            dicts.append(
                {
                    "file_name": f,
                    "image_id": idx,
                }
            )
        return dicts


class BaseTrainer(DefaultTrainer):
    """
    trainer extending DefaultTrainer by an evaluator and a tracking validation loss
    """

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        return COCOEvaluator(dataset_name, output_dir=output_folder)

    def build_hooks(self):
        hooks = super().build_hooks()
        hooks.insert(
            -1,
            LossEvalHook(
                self.cfg.TEST.EVAL_PERIOD,
                self.model,
                build_detection_test_loader(
                    self.cfg,
                    self.cfg.DATASETS.TEST[0],
                    DatasetMapper(
                        self.cfg,
                        True,
                        augmentations=[
                            ResizeShortestEdge(
                                self.cfg.INPUT.MIN_SIZE_TEST,
                                self.cfg.INPUT.MAX_SIZE_TEST,
                            )
                        ],
                    ),
                ),
            ),
        )
        logger = logging.getLogger(__name__)
        logger.critical("actually, this DatasetMapper is used for testing")
        return hooks


class LossEvalHook(HookBase):
    """
    track validation loss, taken from
     https://gist.github.com/ortegatron/c0dad15e49c2b74de8bb09a5615d9f6b
    """

    def __init__(self, eval_period, model, data_loader, name="validation_loss"):
        self._model = model
        self._period = eval_period
        self._data_loader = data_loader
        self._name = name

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
                    f"Loss on {self._name} done {idx + 1}/{total}. {seconds_per_img:.3f} s / img. ETA={str(eta)}",
                    n=5,
                )
            loss_batch = self._get_loss(inputs)
            losses.append(loss_batch)
        mean_loss = np.mean(losses)
        self.trainer.storage.put_scalar(self._name, mean_loss)
        # TODO track different scales
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
