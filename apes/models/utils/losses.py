from mmengine import MODELS
from torch import nn

from ...evaluation.metrics.ChamferDistancePytorch.chamfer3D.dist_chamfer_3D import \
    chamfer_3DDist

# from ...evaluation.metrics.PyTorchEMD.cuda.emd import earth_mover_distance


@MODELS.register_module()
class CrossEntropyLoss(nn.Module):
    def __init__(self, reduction):
        super().__init__()
        self.loss_fn = nn.CrossEntropyLoss(reduction=reduction)

    def forward(self, preds, cls_labels):
        loss = self.loss_fn(preds, cls_labels)
        return loss


@MODELS.register_module()
class ConsistencyLoss(nn.Module):
    def __init__(self, reduction):
        super().__init__()
        self.loss_fn = nn.MSELoss(reduction=reduction)

    def forward(self, tgt):
        loss_list = []
        for i in range(len(tgt)):
            for j in range(len(tgt)):
                if i < j:
                    loss_list.append(self.loss_fn(tgt[i], tgt[j]))
                else:
                    continue
        return sum(loss_list) / len(tgt)


@MODELS.register_module()  # [X] 已新增 CD Loss
class ChamferDistanceLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss_fn = chamfer_3DDist()

    def forward(self, pred_pts, gt_pts):
        dist1, dist2, _, _ = self.loss_fn(pred_pts.transpose(-2,-1).float(), gt_pts.transpose(-2,-1))
        loss = (dist1.mean(axis=1) + dist2.mean(axis=1))
        loss = sum(loss) / len(gt_pts)
        return loss

'''
@MODELS.register_module()
class EarthMoversDistance(nn.Module):
    def __init__(self):
        super().__init__()
        self.emd_function = earth_mover_distance()

    def forward(self, pred_pts, gt_pts):
        dist = self.emd_function(pred_pts.float(), gt_pts)
        loss = sum(dist) / len(gt_pts)
        return loss
'''
