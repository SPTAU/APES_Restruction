from mmengine.structures import BaseDataElement
from torch import Tensor


class ResDataSample(BaseDataElement):

    @property
    def gt_pts(self) -> Tensor:
        return self._gt_pts

    @gt_pts.setter
    def gt_pts(self, value: Tensor) -> None:
        self.set_field(value, '_gt_pts', dtype=Tensor)

    @gt_pts.deleter
    def gt_pts(self) -> None:
        del self._gt_pts

    @property
    def gt_cls_label_onehot(self) -> Tensor:
        return self._gt_cls_label_onehot

    @gt_cls_label_onehot.setter
    def gt_cls_label_onehot(self, value: Tensor) -> None:
        self.set_field(value, '_gt_cls_label_onehot', dtype=Tensor)

    @gt_cls_label_onehot.deleter
    def gt_cls_label_onehot(self) -> None:
        del self._gt_cls_label_onehot

    @property
    def pred_pts(self) -> Tensor:
        return self._pred_pts

    @pred_pts.setter
    def pred_pts(self, value: Tensor) -> None:
        self.set_field(value, '_pred_pts', dtype=Tensor)

    @pred_pts.deleter
    def pred_pts(self) -> None:
        del self._pred_pts
