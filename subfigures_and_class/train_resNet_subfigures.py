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

def train_faster_rcnn_detectron2():
    """
    使用detectron2框架和COCO格式的数据集来训练Faster R-CNN模型。
    """
    print("--- 开始使用 Detectron2 训练 Faster R-CNN 模型 ---")
    
    # --- 1. 定义路径并注册您的COCO数据集 ---
    try:
        # 假设 coco_dataset 文件夹与此脚本在同一级
        script_dir = Path(__file__).resolve().parent
    except NameError:
        script_dir = Path.cwd()
    
    COCO_DATASET_PATH = script_dir / "coco_dataset"
    TRAIN_JSON = COCO_DATASET_PATH / "annotations" / "train.json"
    TRAIN_IMAGES = COCO_DATASET_PATH / "images" / "train"
    VAL_JSON = COCO_DATASET_PATH / "annotations" / "val.json"
    VAL_IMAGES = COCO_DATASET_PATH / "images" / "val"
    
    # 注册数据集，让Detectron2知道去哪里找数据
    # 我们为训练集和验证集分别注册
    register_coco_instances("subfigure_train", {}, TRAIN_JSON, TRAIN_IMAGES)
    register_coco_instances("subfigure_val", {}, VAL_JSON, VAL_IMAGES)
    
    print("数据集注册成功。")

    # --- 2. 配置模型 ---
    cfg = get_cfg()
    
    # 从detectron2的模型库加载一个标准的Faster R-CNN配置
    # 这个模型使用ResNet-50作为主干网络，并经过了3倍时长的训练
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_152_FPN_3x.yaml"))
    
    # 设置数据集
    cfg.DATASETS.TRAIN = ("subfigure_train",)
    cfg.DATASETS.TEST = ("subfigure_val",)
    
    # 设置训练参数
    cfg.DATALOADER.NUM_WORKERS = 2
    # 加载预训练模型的权重
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_152_FPN_3x.yaml")
    cfg.SOLVER.IMS_PER_BATCH = 2  # 根据您的GPU显存调整，如果显存不足，可以设为1
    cfg.SOLVER.BASE_LR = 0.0025   # 学习率
    cfg.SOLVER.MAX_ITER = 1500   # 总训练迭代次数。对于微调，1500-3000是一个不错的起点
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # 您的数据集中只有'subfigure'这一个类别
    
    # 设置输出目录
    cfg.OUTPUT_DIR = "./runs/faster_rcnn_detectron2_output"
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    
    print("模型配置完成。")

    # --- 3. 开始训练 ---
    # 使用Detectron2的标准训练器
    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=False)
    print("开始训练...")
    trainer.train()
    print("训练完成！")

    # --- 4. 评估模型 ---
    print("\n--- 开始在验证集上评估模型 ---")
    # 加载训练好的最佳模型
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
    # 设置一个合理的预测置信度阈值
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7  
    
    # 创建评估器
    evaluator = COCOEvaluator("subfigure_val", output_dir=cfg.OUTPUT_DIR)
    # 创建验证集的数据加载器
    val_loader = build_detection_test_loader(cfg, "subfigure_val")
    
    # 在验证集上运行推理并获取结果
    results = inference_on_dataset(trainer.model, val_loader, evaluator)
    print("评估完成。评估结果如下：")
    # COCOEvaluator会自动打印出所有指标，包括 P, R, mAP50, 和 mAP50-95
    print(results) 

if __name__ == '__main__':
    train_faster_rcnn_detectron2()

