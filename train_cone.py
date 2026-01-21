from ultralytics import YOLO
import matplotlib.pyplot as plt
import os

# 设置为 True 可以看到更详细的每层结构
VERBOSE = True 

def train_hardcore():
    # 1. 加载模型
    model = YOLO('yolov8n.pt') 

    # 打印网络结构 (满足你对"神经网络呢"的疑问)
    if VERBOSE:
        print("\n=== 神经网络架构 (Backbone + Neck + Head) ===")
        # model.model 是底层的 PyTorch nn.Module 对象
        print(model.model) 
        print("===========================================\n")

    # 2. 开始训练 (包含梯度下降过程)
    # 我们开启 plots=True，这样训练完会自动生成 Loss 曲线图
    results = model.train(
        data='data.yaml',
        epochs=50,          # 演示用50轮，实际建议100
        imgsz=640,
        batch=16,
        workers=0,          # Windows 必须为 0
        name='cone_hardcore',
        amp=False,
        # === 核心超参数 (控制梯度下降) ===
        optimizer='AdamW',  # 显式指定优化器 (SGD, Adam, AdamW)
        lr0=0.01,           # 初始学习率 (步长)
        momentum=0.937,     # 动量 (让下坡更有惯性)
        cos_lr=True,        # 使用余弦退火调整学习率
        
        # === 输出控制 ===
        plots=True,         # 自动画出 Loss 曲线和 mAP 曲线
        verbose=True        # 打印每一个 Epoch 的详细 Loss
    )

    print("\n✅ 训练结束！")
    print(f"结果保存在: {results.save_dir}")
    print("请去该文件夹查看 'results.png'，那是你的 Loss 下降曲线和准确率曲线。")

if __name__ == '__main__':
    train_hardcore()