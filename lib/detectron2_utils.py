import datetime
import logging
import os
import time

import numpy as np
import torch
from detectron2.data import DatasetMapper, build_detection_test_loader
from detectron2.data import detection_utils
from detectron2.data.transforms import ResizeShortestEdge
from detectron2.engine.defaults import DefaultTrainer
from detectron2.engine.hooks import HookBase
from detectron2.evaluation import COCOEvaluator
from detectron2.utils import comm
from detectron2.utils.logger import log_every_n_seconds
from detectron2.utils.visualizer import Visualizer
from google.colab.patches import cv2_imshow


def visualize_detectron2_loader(data_loader, cfg, n_examples=5):
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
            visualizer = Visualizer(img, scale=1)
            target_fields = per_image["instances"].get_fields()
            vis = visualizer.overlay_instances(
                boxes=target_fields.get("gt_boxes", None),
                masks=target_fields.get("gt_masks", None),
            )
            cv2_imshow(vis.get_image()[:, :, ::-1])
            n_examples -= 1


def create_output(cfg):
    """
    creates detectron compatible output dir from cfg
    """
    dirname = cfg.OUTPUT_DIR
    os.makedirs(dirname, exist_ok=True)
    open(os.path.join(dirname, "metrics.json"), "a").close()


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
