_base_ = ['../_base_/models/apes_res_global.py',
          '../_base_/datasets/modelnet_res.py',
          '../_base_/schedules/schedule_200epochs.py',
          '../_base_/default_runtime.py']

experiment_name = '{{fileBasenameNoExtension}}'  # use cfg file name as exp name
work_dir = f'./work_dirs/{experiment_name}'  # working dir to save ckpts and logs
visualizer = dict(vis_backends=[dict(type='ModifiedLocalVisBackend')])
