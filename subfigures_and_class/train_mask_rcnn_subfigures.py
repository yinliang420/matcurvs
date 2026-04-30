import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

import os
from pathlib import Path
from detectron2.model_zoo import model_zoo
from detectron2.data.datasets import register_coco_instances
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader

def train_mask_rcnn():
    """
    使用COCO格式的数据集训练Mask R-CNN模型。
    """
    print("--- 开始训练 Mask R-CNN 模型 ---")
    
    # --- 1. 定义路径并注册数据集 ---
    try:
        script_dir = Path(__file__).resolve().parent
    except NameError:
        script_dir = Path.cwd()
    
    COCO_DATASET_PATH = script_dir / "coco_dataset"
    TRAIN_JSON = COCO_DATASET_PATH / "annotations" / "train.json"
    TRAIN_IMAGES = COCO_DATASET_PATH / "images" / "train"
    VAL_JSON = COCO_DATASET_PATH / "annotations" / "val.json"
    VAL_IMAGES = COCO_DATASET_PATH / "images" / "val"
    
    # 注册数据集，让Detectron2知道去哪里找数据
    register_coco_instances("subfigure_train", {}, TRAIN_JSON, TRAIN_IMAGES)
    register_coco_instances("subfigure_val", {}, VAL_JSON, VAL_IMAGES)
    
    print("数据集注册成功。")

    # --- 2. 配置模型 ---
    cfg = get_cfg()
    # 我们使用一个标准的、预训练好的R50-FPN检测模型
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml"))
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml")
    
    # 设置数据集
    cfg.DATASETS.TRAIN = ("subfigure_train",)
    cfg.DATASETS.TEST = ("subfigure_val",)
    
    # 设置训练参数
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.SOLVER.IMS_PER_BATCH = 2  # 根据您的GPU显存调整
    cfg.SOLVER.BASE_LR = 0.0025
    cfg.SOLVER.MAX_ITER = 1500  # 训练迭代次数
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # 只有'subfigure'这一个类别
    
    # 设置输出目录
    cfg.OUTPUT_DIR = "./runs/maskrcnn_output"
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    
    print("模型配置完成。")

    # --- 3. 开始训练 ---
    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=False)
    print("开始训练...")
    trainer.train()
    print("训练完成！")

    # --- 4. 评估模型 ---
    print("\n--- 开始在验证集上评估模型 ---")
    # 加载训练好的最佳模型
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7  # 设置一个合理的阈值
    
    evaluator = COCOEvaluator("subfigure_val", output_dir=cfg.OUTPUT_DIR)
    val_loader = build_detection_test_loader(cfg, "subfigure_val")
    
    results = inference_on_dataset(trainer.model, val_loader, evaluator)
    print("评估完成。结果如下：")
    print(results) # COCOEvaluator会打印出所有指标，包括 P, R, mAP50, mAP50-95

if __name__ == '__main__':
    train_mask_rcnn()
