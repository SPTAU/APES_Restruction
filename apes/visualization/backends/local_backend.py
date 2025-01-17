import os

import matplotlib.pyplot as plt
import numpy as np
from mmengine.registry import VISBACKENDS
from mmengine.visualization.vis_backend import LocalVisBackend, force_init_env


@VISBACKENDS.register_module()
class ModifiedLocalVisBackend(LocalVisBackend):

    """this backend won't save config file and any metric values. it is used for saving images only."""

    def add_config(self, config, **kwargs):
        pass

    def add_scalars(self, scalar_dict, step=0, file_path=None, **kwargs):
        pass

    @force_init_env
    def add_image(self, name, pcd: np.ndarray, **kwargs):
        # pcd.shape == (N, C), where C: x y z r g b
        os.makedirs(self._img_save_dir, exist_ok=True)
        saved_path = os.path.join(self._img_save_dir, f'{name}.png')
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.set_xlim3d(-0.6, 0.6)
        ax.set_ylim3d(-0.6, 0.6)
        ax.set_zlim3d(-0.6, 0.6)
        ax.scatter(pcd[:, 0], pcd[:, 2], pcd[:, 1], c=pcd[:, 3:]/255., marker='o', s=2)
        plt.axis('off')
        plt.grid('off')
        plt.savefig(saved_path, bbox_inches='tight')
        plt.close(fig)

@VISBACKENDS.register_module()
class ResLocalVisBackend(LocalVisBackend):

    """this backend won't save config file and any metric values. it is used for saving images only."""

    def add_config(self, config, **kwargs):
        pass

    def add_scalars(self, scalar_dict, step=0, file_path=None, **kwargs):
        pass

    @force_init_env
    def add_image(self, name, pcd: np.ndarray, **kwargs):
        # pcd.shape == (N, C), where C: x y z r g b
        if '/' in name:
            sub_folder = name.split('/')[0]
            os.makedirs(os.path.join(self._img_save_dir, sub_folder), exist_ok=True)
        else:
            os.makedirs(self._img_save_dir, exist_ok=True)
        saved_path = os.path.join(self._img_save_dir, f'{name}.png')
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.set_xlim3d(-0.6, 0.6)
        ax.set_ylim3d(-0.6, 0.6)
        ax.set_zlim3d(-0.6, 0.6)
        ax.scatter(pcd[:, 0], pcd[:, 2], pcd[:, 1], c='r', label='gt_pts', marker='o', s=2)
        ax.scatter(pcd[:, 3], pcd[:, 5], pcd[:, 4], c='g', label='pred_pts', marker='o', s=2)
        plt.axis('off')
        plt.grid('off')
        plt.savefig(saved_path, bbox_inches='tight')
        plt.close(fig)

    @force_init_env
    def add_single_image(self, name, pcd: np.ndarray, **kwargs):
        # pcd.shape == (N, C), where C: x y z r g b
        if '/' in name:
            sub_folder = name.split('/')[0]
            os.makedirs(os.path.join(self._img_save_dir, sub_folder), exist_ok=True)
        else:
            os.makedirs(self._img_save_dir, exist_ok=True)
        saved_path = os.path.join(self._img_save_dir, f'{name}.png')
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.set_xlim3d(-0.6, 0.6)
        ax.set_ylim3d(-0.6, 0.6)
        ax.set_zlim3d(-0.6, 0.6)
        ax.scatter(pcd[:, 3], pcd[:, 5], pcd[:, 4], c='g', label='pred_pts', marker='o', s=2)
        plt.axis('off')
        plt.grid('off')
        plt.savefig(saved_path, bbox_inches='tight')
        plt.close(fig)
