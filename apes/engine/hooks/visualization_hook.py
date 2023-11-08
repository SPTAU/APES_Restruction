import os
from typing import Dict, List

import numpy as np
import torch
from einops import pack, rearrange, repeat
from mmengine.hooks import Hook
from mmengine.registry import HOOKS

from ...structures.cls_data_sample import ClsDataSample
from ...structures.res_data_sample import ResDataSample
from ...structures.seg_data_sample import SegDataSample


@HOOKS.register_module()
class CLSVisualizationHook(Hook):
    def after_test_iter(self, runner, batch_idx: int, data_batch: Dict=None, outputs: List[ClsDataSample]=None):
        if runner.world_size > 1:
            ds1_idx = runner.model.module.backbone.ds1.idx.cpu()
            ds2_idx = torch.gather(ds1_idx, dim=1, index=runner.model.module.backbone.ds2.idx.cpu())
        else:
            ds1_idx = runner.model.backbone.ds1.idx.cpu()
            ds2_idx = torch.gather(ds1_idx, dim=1, index=runner.model.backbone.ds2.idx.cpu())
        inputs, data_samples = data_batch['inputs'], data_batch['data_samples']
        inputs = rearrange(inputs, 'B C N -> B N C')
        bg_color = repeat(torch.tensor([192., 192., 192.]), 'C -> B N C', B=inputs.shape[0], N=inputs.shape[1])
        red = torch.tensor([255., 0., 0.])
        ds1_rgb = torch.scatter(bg_color, dim=1, index=repeat(ds1_idx, 'B N -> B N C', C=3), src=repeat(red, 'C -> B N C', B=ds1_idx.shape[0], N=ds1_idx.shape[1]))
        ds2_rgb = torch.scatter(bg_color, dim=1, index=repeat(ds2_idx, 'B N -> B N C', C=3), src=repeat(red, 'C -> B N C', B=ds2_idx.shape[0], N=ds2_idx.shape[1]))
        ds1_xyz_rgb, _ = pack([inputs, ds1_rgb], 'B N *')
        ds2_xyz_rgb, _ = pack([inputs, ds2_rgb], 'B N *')
        for i, (ds1, ds2) in enumerate(zip(ds1_xyz_rgb, ds2_xyz_rgb)):
            runner.visualizer.add_image(f'cls_pcd{i+runner.test_dataloader.batch_size*(batch_idx*runner.world_size+runner.rank)}_ds1', ds1.numpy())
            runner.visualizer.add_image(f'cls_pcd{i+runner.test_dataloader.batch_size*(batch_idx*runner.world_size+runner.rank)}_ds2', ds2.numpy())


@HOOKS.register_module()
class SEGVisualizationHook(Hook):
    def after_test_iter(self, runner, batch_idx: int, data_batch: Dict=None, outputs: List[SegDataSample]=None):
        inputs = rearrange(data_batch['inputs'], 'B C N -> B N C').cpu().numpy()
        for i, (xyz, output) in enumerate(zip(inputs, outputs)):
            palette = np.array(output.metainfo['palette'])
            rgb = palette[output.pred_seg_label.long().cpu().numpy()]
            xyz_rgb, _ = pack([xyz, rgb], 'N *')
            runner.visualizer.add_image(f'seg_pcd{i+runner.test_dataloader.batch_size*(batch_idx*runner.world_size+runner.rank)}', xyz_rgb)

@HOOKS.register_module()
class RESVisualizationHook(Hook):
    def after_test_iter(self, runner, batch_idx: int, data_batch: Dict=None, outputs: List[ResDataSample]=None) -> None:
        inputs = rearrange(data_batch['inputs'], 'B C N -> B N C').cpu().numpy()
        for i, output in enumerate(outputs):
            idx = i+runner.test_dataloader.batch_size*(batch_idx*runner.world_size+runner.rank)
            gt_xyz = output.gt_pts.transpose(0, 1).cpu().numpy()
            pred_xyz = output.pred_pts.transpose(0, 1).cpu().numpy()
            gt_pred_xyz, _ = pack([gt_xyz, pred_xyz], 'N *')
            runner.visualizer.add_image(f'res_pcd{idx}', gt_pred_xyz)
            # runner.visualizer.add_single_image(f'res_pcd{idx}', gt_pred_xyz)

    def after_val_iter(self, runner, batch_idx: int, data_batch: Dict=None, outputs: List[ResDataSample]=None) -> None:
        if runner.epoch % 10 == 0:
            # os.makedirs(os.path.join(runner.visualizer._img_save_dir, runner.epoch), exist_ok=True)
            inputs = rearrange(data_batch['inputs'], 'B C N -> B N C').cpu().numpy()
            for i, output in enumerate(outputs):
                idx = i + runner.val_dataloader.batch_size * (batch_idx * runner.world_size + runner.rank)
                if idx < 10 and idx >=0:
                    gt_xyz = output.gt_pts.transpose(0, 1).cpu().numpy()
                    pred_xyz = output.pred_pts.transpose(0, 1).cpu().numpy()
                    gt_pred_xyz, _ = pack([gt_xyz, pred_xyz], 'N *')
                    runner.visualizer.add_image(f'epoch{runner.epoch}/res_pcd{idx}', gt_pred_xyz)
