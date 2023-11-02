import torch
from mmengine.evaluator import BaseMetric
from mmengine.registry import METRICS

from .ChamferDistancePytorch.chamfer3D.dist_chamfer_3D import chamfer_3DDist


@METRICS.register_module()
class ChamferDistance(BaseMetric):
    def __init__(self, mode='val'):
        super(ChamferDistance, self).__init__()
        self.mode = mode
        self.cd_function = chamfer_3DDist()

    default_prefix = 'Restuctation'

    def process(self, inputs, data_samples: list[dict]):  # data_samples is a List of Dict, not a List of ResDataSample
        for data_sample in data_samples:
            result = dict(gt_pts=data_sample['gt_pts'], pred_pts=data_sample['pred_pts'])
            self.results.append(result)  # self.results is actually the 'results' in the compute_metrics method

    def compute_metrics(self, results) -> dict:
        pred_pts_tensor = torch.stack([result['pred_pts'] for result in results])
        gt_pts_tensor = torch.stack([result['gt_pts'] for result in results])
        pred_pts = pred_pts_tensor.transpose(-2,-1).cuda()
        gt_pts = gt_pts_tensor.transpose(-2,-1).float().cuda()
        dist1, dist2, _, _ = self.cd_function(pred_pts, gt_pts)
        chamferDistance = (dist1.mean(axis=1) + dist2.mean(axis=1))
        chamferDistance = torch.sum(chamferDistance) / gt_pts.shape[0]
        if self.mode == 'val':
            return dict(val_chamfer_dist=chamferDistance)
        elif self.mode == 'test':
            return dict(test_chamfer_dist=chamferDistance)
        else:
            raise RuntimeError(f'Invalid mode "{self.mode}". Only supports val and test mode')
