# Ultralytics YOLO 🚀, AGPL-3.0 license

import torch
import torchvision

from ultralytics.data import ClassificationDataset, build_dataloader
from ultralytics.engine.trainer import BaseTrainer
from ultralytics.models import yolo
from ultralytics.nn.tasks import ClassificationModel, attempt_load_one_weight
from ultralytics.utils import DEFAULT_CFG, LOGGER, RANK, colorstr
from ultralytics.utils.plotting import plot_images, plot_results
from ultralytics.utils.torch_utils import is_parallel, strip_optimizer, torch_distributed_zero_first

def main():
    print("STH", sth)
    cfg =    {"task": "detect", "mode": "train", "model": null, "data": null, "epochs": 100, "patience": 50, "batch": 16, "imgsz": 640, "save": true, "save_period": -1, "cache": false, "device": null, "workers": 8, "project": null, "name": null, "exist_ok": false, "pretrained": true, "optimizer": "auto", "verbose": true, "seed": 0, "deterministic": true, "single_cls": false, "rect": false, "cos_lr": false, "close_mosaic": 10, "resume": false, "amp": true, "fraction": 1.0, "profile": false, "freeze": null, "overlap_mask": true, "mask_ratio": 4, "dropout": 0.0, "val": true, "split": "val", "save_json": false, "save_hybrid": false, "conf": null, "iou": 0.7, "max_det": 300, "half": false, "dnn": false, "plots": true, "source": null, "show": false, "save_txt": false, "save_conf": false, "save_crop": false, "show_labels": true, "show_conf": true, "vid_stride": 1, "stream_buffer": false, "line_width": null, "visualize": false, "augment": false, "agnostic_nms": false, "classes": null, "retina_masks": false, "boxes": true, "format": "torchscript", "keras": false, "optimize": false, "int8": false, "dynamic": false, "simplify": false, "opset": null, "workspace": 4, "nms": false, "lr0": 0.01, "lrf": 0.01, "momentum": 0.937, "weight_decay": 0.0005, "warmup_epochs": 3.0, "warmup_momentum": 0.8, "warmup_bias_lr": 0.1, "box": 7.5, "cls": 0.5, "dfl": 1.5, "pose": 12.0, "kobj": 1.0, "label_smoothing": 0.0, "nbs": 64, "hsv_h": 0.015, "hsv_s": 0.7, "hsv_v": 0.4, "degrees": 0.0, "translate": 0.1, "scale": 0.5, "shear": 0.0, "perspective": 0.0, "flipud": 0.0, "fliplr": 0.5, "mosaic": 1.0, "mixup": 0.0, "copy_paste": 0.0, "cfg": null, "tracker": "botsort.yaml"}
    overrides = {"model": "yolov8n-cls.yaml", "task": "classify", "data": "mnist160", "epochs": 100, "imgsz": 64, "mode": "train"}
    import torch.distributed as dist
    import torch_xla.experimental.pjrt_backend
    dist.init_process_group("xla", init_method="pjrt://")
    ClassificationTrainer(cfg=cfg, overrides=overrides)._do_train(world_size)

if __name__=="__main__":
    main()

class ClassificationTrainer(BaseTrainer):
    """
    A class extending the BaseTrainer class for training based on a classification model.

    Notes:
        - Torchvision classification models can also be passed to the 'model' argument, i.e. model='resnet18'.

    Example:
        ```python
        from ultralytics.models.yolo.classify import ClassificationTrainer

        args = dict(model='yolov8n-cls.pt', data='imagenet10', epochs=3)
        trainer = ClassificationTrainer(overrides=args)
        trainer.train()
        ```
    """

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        """Initialize a ClassificationTrainer object with optional configuration overrides and callbacks."""
        if overrides is None:
            overrides = {}
        overrides['task'] = 'classify'
        if overrides.get('imgsz') is None:
            overrides['imgsz'] = 224
        super().__init__(cfg, overrides, _callbacks)

    def set_model_attributes(self):
        """Set the YOLO model's class names from the loaded dataset."""
        self.model.names = self.data['names']

    def get_model(self, cfg=None, weights=None, verbose=True):
        """Returns a modified PyTorch model configured for training YOLO."""
        model = ClassificationModel(cfg, nc=self.data['nc'], verbose=verbose and RANK == -1)
        if weights:
            model.load(weights)

        for m in model.modules():
            if not self.args.pretrained and hasattr(m, 'reset_parameters'):
                m.reset_parameters()
            if isinstance(m, torch.nn.Dropout) and self.args.dropout:
                m.p = self.args.dropout  # set dropout
        for p in model.parameters():
            p.requires_grad = True  # for training
        return model
        
    def _do_xla_ddp(self, world_size):
        import torch_xla.distributed.xla_multiprocessing as xmp
        xmp.spawn(spawn_train, args=(self.xla_save_args[0], self.xla_save_args[1], world_size))

    def setup_model(self):
        """Load, create or download model for any task."""
        if isinstance(self.model, torch.nn.Module):  # if model is loaded beforehand. No setup needed
            return

        model, ckpt = str(self.model), None
        # Load a YOLO model locally, from torchvision, or from Ultralytics assets
        if model.endswith('.pt'):
            self.model, ckpt = attempt_load_one_weight(model, device='cpu')
            for p in self.model.parameters():
                p.requires_grad = True  # for training
        elif model.split('.')[-1] in ('yaml', 'yml'):
            self.model = self.get_model(cfg=model)
        elif model in torchvision.models.__dict__:
            self.model = torchvision.models.__dict__[model](weights='IMAGENET1K_V1' if self.args.pretrained else None)
        else:
            FileNotFoundError(f'ERROR: model={model} not found locally or online. Please check model name.')
        ClassificationModel.reshape_outputs(self.model, self.data['nc'])

        return ckpt

    def build_dataset(self, img_path, mode='train', batch=None):
        return ClassificationDataset(root=img_path, args=self.args, augment=mode == 'train', prefix=mode)

    def get_dataloader(self, dataset_path, batch_size=16, rank=0, mode='train'):
        """Returns PyTorch DataLoader with transforms to preprocess images for inference."""
        with torch_distributed_zero_first(rank):  # init dataset *.cache only once if DDP
            dataset = self.build_dataset(dataset_path, mode)

        loader = build_dataloader(dataset, batch_size, self.args.workers, rank=rank)
        # Attach inference transforms
        if mode != 'train':
            if is_parallel(self.model):
                self.model.module.transforms = loader.dataset.torch_transforms
            else:
                self.model.transforms = loader.dataset.torch_transforms
        return loader

    def preprocess_batch(self, batch):
        """Preprocesses a batch of images and classes."""
        batch['img'] = batch['img'].to(self.device)
        batch['cls'] = batch['cls'].to(self.device)
        return batch

    def progress_string(self):
        """Returns a formatted string showing training progress."""
        return ('\n' + '%11s' * (4 + len(self.loss_names))) % \
            ('Epoch', 'GPU_mem', *self.loss_names, 'Instances', 'Size')

    def get_validator(self):
        """Returns an instance of ClassificationValidator for validation."""
        self.loss_names = ['loss']
        return yolo.classify.ClassificationValidator(self.test_loader, self.save_dir)

    def label_loss_items(self, loss_items=None, prefix='train'):
        """
        Returns a loss dict with labelled training loss items tensor. Not needed for classification but necessary for
        segmentation & detection
        """
        keys = [f'{prefix}/{x}' for x in self.loss_names]
        if loss_items is None:
            return keys
        loss_items = [round(float(loss_items), 5)]
        return dict(zip(keys, loss_items))

    def plot_metrics(self):
        """Plots metrics from a CSV file."""
        plot_results(file=self.csv, classify=True, on_plot=self.on_plot)  # save results.png

    def final_eval(self):
        """Evaluate trained model and save validation results."""
        for f in self.last, self.best:
            if f.exists():
                strip_optimizer(f)  # strip optimizers
                if f is self.best:
                    LOGGER.info(f'\nValidating {f}...')
                    self.validator.args.data = self.args.data
                    self.validator.args.plots = self.args.plots
                    self.metrics = self.validator(model=f)
                    self.metrics.pop('fitness', None)
                    self.run_callbacks('on_fit_epoch_end')
        LOGGER.info(f"Results saved to {colorstr('bold', self.save_dir)}")

    def plot_training_samples(self, batch, ni):
        """Plots training samples with their annotations."""
        plot_images(
            images=batch['img'],
            batch_idx=torch.arange(len(batch['img'])),
            cls=batch['cls'].view(-1),  # warning: use .view(), not .squeeze() for Classify models
            fname=self.save_dir / f'train_batch{ni}.jpg',
            on_plot=self.on_plot)

