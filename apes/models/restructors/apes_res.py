from mmengine.registry import MODELS
from mmengine.model import BaseModel
from typing import List
from torch import Tensor
from ...structures.res_data_sample import ResDataSample
import torch
from einops import pack


@MODELS.register_module()
class APESRestructor(BaseModel):
    def __init__(self,
                 backbone: dict,
                 neck: dict = None,
                 head: dict = None,
                 data_preprocessor: dict = None,
                 init_cfg: List[dict] = None):
        # 20231025已修改
        super(APESRestructor, self).__init__(data_preprocessor, init_cfg)
        self.backbone = MODELS.build(backbone)  # [ ] 考虑切断中间点坐标的 skip connect
        self.neck = MODELS.build(neck) if neck is not None else None
        self.head = MODELS.build(head)
        self.cd_loss = MODELS.build(dict(type='ChamferDistanceLoss'))  # [X] 修改LOSS 为 CDLoss

    def forward(self, inputs: Tensor, data_samples: List[ResDataSample], mode: str):
        # 不需要修改
        if mode == 'loss':
            return self.loss(inputs, data_samples)
        elif mode == 'predict':
            return self.predict(inputs, data_samples)
        elif mode == 'tensor':
            return self.tensor(inputs, data_samples)
        else:
            raise RuntimeError(f'Invalid mode "{mode}". Only supports loss, predict and tensor mode')

    def loss(self, inputs: Tensor, data_samples: List[ResDataSample]) -> dict:
        # 20231026已修改
        gt_pts = self.get_gt_pts(data_samples)
        gt_cls_labels_onehot= self.get_gt_labels_onehot(data_samples)
        losses = dict()
        x = self.extract_features(inputs, gt_cls_labels_onehot)
        pred_pts = self.head(x)
        cd_loss = self.cd_loss(pred_pts, gt_pts)
        losses.update(dict(loss=cd_loss))
        return losses

    def predict(self, inputs: Tensor, data_samples: List[ResDataSample]) -> List[ResDataSample]:
        data_samples_list = []
        gt_pts_lists = self.get_gt_pts(data_samples)
        gt_cls_labels_onehot= self.get_gt_labels_onehot(data_samples)
        x = self.extract_features(inputs, gt_cls_labels_onehot)
        pred_pts_lists = self.head(x)  # [X] 修改 head
        for data_sample, pred_pts_list in zip(data_samples, pred_pts_lists):
            data_sample.pred_pts = pred_pts_list
            data_samples_list.append(data_sample)
        return data_samples_list

    def tensor(self, inputs: Tensor, data_samples: List[ResDataSample]) -> Tensor:
        gt_pts = self.get_gt_pts(data_samples)
        gt_cls_labels_onehot= self.get_gt_labels_onehot(data_samples)
        x = self.extract_features(inputs, gt_cls_labels_onehot)
        pred_pts = self.head(x)
        return pred_pts

    @staticmethod
    def get_gt_pts(data_samples: List[ResDataSample]) -> Tensor:
        # 20231026已修改
        pts_list = []
        for data_sample in data_samples:
            pts_list.append(data_sample.gt_pts)  # [X] 修改 datasample，增加 gt_pts
        pts, _ = pack(pts_list, '* C N')  # shape == (B, C, N)
        return pts

    @staticmethod
    def get_gt_labels_onehot(data_samples: List[ResDataSample]) -> Tensor:
        cls_labels_list = []
        for data_sample in data_samples:
            assert data_sample.gt_cls_label_onehot is not None
            cls_labels_list.append(data_sample.gt_cls_label_onehot)
        cls_labels, _ = pack(cls_labels_list, '* C N')  # shape == (B, C, N=1)
        return cls_labels

    def extract_features(self, inputs, shape_classes) -> Tensor:
        x = self.backbone(inputs, shape_classes)
        if self.neck is not None:
            x = self.neck(x)
        return x
