"""
defines folder structure of the project. Adapt if structure need to be changed.
"""

import os

project_dir = ".."

path_to_data_dir = os.path.join(project_dir, "data")

path_to_output_dir = os.path.join(project_dir, "output")

path_to_imgs_dir = os.path.join(path_to_data_dir, "imgs")

path_to_masks_dir = os.path.join(path_to_data_dir, "masks")

path_to_anno_dir = os.path.join(path_to_data_dir, "anno")

path_to_masked_objs_dir = os.path.join(path_to_data_dir, "masked_objs")

