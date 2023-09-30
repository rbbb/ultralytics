from ultralytics.models.yolo.classify.train import ClassificationTrainer

def main():
    true=True
    false=False
    null=None
    cfg = {"task": "detect", "mode": "train", "model": null, "data": null, "epochs": 100, "patience": 50, "batch": 16, "imgsz": 640, "save": true, "save_period": -1, "cache": false, "device": null, "workers": 8, "project": null, "name": null, "exist_ok": false, "pretrained": true, "optimizer": "auto", "verbose": true, "seed": 0, "deterministic": true, "single_cls": false, "rect": false, "cos_lr": false, "close_mosaic": 10, "resume": false, "amp": true, "fraction": 1.0, "profile": false, "freeze": null, "overlap_mask": true, "mask_ratio": 4, "dropout": 0.0, "val": true, "split": "val", "save_json": false, "save_hybrid": false, "conf": null, "iou": 0.7, "max_det": 300, "half": false, "dnn": false, "plots": true, "source": null, "show": false, "save_txt": false, "save_conf": false, "save_crop": false, "show_labels": true, "show_conf": true, "vid_stride": 1, "stream_buffer": false, "line_width": null, "visualize": false, "augment": false, "agnostic_nms": false, "classes": null, "retina_masks": false, "boxes": true, "format": "torchscript", "keras": false, "optimize": false, "int8": false, "dynamic": false, "simplify": false, "opset": null, "workspace": 4, "nms": false, "lr0": 0.01, "lrf": 0.01, "momentum": 0.937, "weight_decay": 0.0005, "warmup_epochs": 3.0, "warmup_momentum": 0.8, "warmup_bias_lr": 0.1, "box": 7.5, "cls": 0.5, "dfl": 1.5, "pose": 12.0, "kobj": 1.0, "label_smoothing": 0.0, "nbs": 64, "hsv_h": 0.015, "hsv_s": 0.7, "hsv_v": 0.4, "degrees": 0.0, "translate": 0.1, "scale": 0.5, "shear": 0.0, "perspective": 0.0, "flipud": 0.0, "fliplr": 0.5, "mosaic": 1.0, "mixup": 0.0, "copy_paste": 0.0, "cfg": null, "tracker": "botsort.yaml", "device":"xla"}
    overrides = {"model": "yolov8n-cls.yaml", "task": "classify", "data": "cifar100", "epochs": 100, "imgsz": 64, "mode": "train", "device":"xla"}
    import torch.distributed as dist
    import torch_xla.experimental.pjrt_backend
    dist.init_process_group("xla", init_method="pjrt://")
    ClassificationTrainer(cfg=cfg, overrides=overrides)._do_train(8)

if __name__=="__main__":
    main()
