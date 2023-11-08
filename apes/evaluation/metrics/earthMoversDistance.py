import torch
from mmengine.evaluator import BaseMetric
from mmengine.registry import METRICS

from .PyTorchEMD.cuda.emd import earth_mover_distance


@METRICS.register_module()
class EarthMoversDistance(BaseMetric):
    def __init__(self, mode='val'):
        super(EarthMoversDistance, self).__init__()
        self.mode = mode
        self.emd_function = earth_mover_distance()

    default_prefix = 'Restuctation'

    def process(self, inputs, data_samples: list[dict]):  # data_samples is a List of Dict, not a List of ResDataSample
        for data_sample in data_samples:
            result = dict(gt_pts=data_sample['gt_pts'], pred_pts=data_sample['pred_pts'])
            self.results.append(result)  # self.results is actually the 'results' in the compute_metrics method

    def compute_metrics(self, results) -> dict:
        pred_pts_tensor = torch.stack([result['pred_pts'] for result in results])
        gt_pts_tensor = torch.stack([result['gt_pts'] for result in results])
        pred_pts = pred_pts_tensor.cuda()
        gt_pts = gt_pts_tensor.float().cuda()
        dist = self.emd_function(pred_pts, gt_pts)
        earthMoversDistance = torch.sum(dist) / gt_pts.shape[0]
        if self.mode == 'val':
            return dict(val_chamfer_dist=earthMoversDistance)
        elif self.mode == 'test':
            return dict(test_chamfer_dist=earthMoversDistance)
        else:
            raise RuntimeError(f'Invalid mode "{self.mode}". Only supports val and test mode')
