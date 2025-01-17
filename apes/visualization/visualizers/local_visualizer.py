from mmengine.registry import VISUALIZERS
from mmengine.visualization import Visualizer


@VISUALIZERS.register_module()
class APESVisualizer(Visualizer):
    def add_image(self, name, pcd) -> None:
        for vis_backend in self._vis_backends.values():
            vis_backend.add_image(name, pcd)

    def add_single_image(self, name, pcd) -> None:
        for vis_backend in self._vis_backends.values():
            vis_backend.add_single_image(name, pcd)
