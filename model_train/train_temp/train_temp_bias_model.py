import argparse
import os.path
import sys
from pathlib import Path

# 获取当前文件的绝对路径
FILE = Path(__file__).resolve()
# 获取项目根目录（假设脚本在项目子目录中）
ROOT = FILE.parents[0]  # 项目根目录
# 添加项目根目录到 Python 路径
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
# 设置工作目录为项目根目录
os.chdir(ROOT)


import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from datasets import   FaceData
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from models.temp_biase.temp_bias import  CrossAttentionRegressionModel
import sys
# print(sys.path)
# 训练函数
# 更高级的版本，包含更多指标和可视化
def train_model(model, train_loader, val_loader, num_epochs=500, lr=1e-3, early_stopping_patience=20):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    model.to(device)
    criterion = nn.MSELoss()
    # criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    history = {
        'train_loss': [], 'val_loss': [], 'learning_rate': [],
        'train_mae': [], 'val_mae': [], 'epoch_times': []
    }

    best_val_loss = float('inf')
    best_model_weights = None
    patience_counter = 0
    # 在训练开始前初始化GradScaler
    scaler = torch.amp.GradScaler("cuda")
    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        # 在训练开始前初始化GradScaler
        # 训练阶段
        model.train()
        train_loss = 0.0
        train_mae = 0.0
        train_mae_max = 0.
        train_pbar = tqdm(train_loader, desc=f'🏃 训练 Epoch {epoch + 1}/{num_epochs}',
                          bar_format='{l_bar}{bar:25}{r_bar}{bar:-10b}')

        for batch_idx, (images, extra_features, labels) in enumerate(train_pbar):
            images, extra_features, labels = images.to(device), extra_features.to(device), labels.to(device)
            optimizer.zero_grad()
            # 使用混合精度训练
            with torch.amp.autocast('cuda'):
                outputs = model(images, extra_features)
                loss = criterion(outputs, labels)
                mae = torch.mean(torch.abs(outputs - labels))
                train_mae_max = max(train_mae_max, torch.abs(outputs - labels).max())

            # 使用scaler进行反向传播
            scaler.scale(loss).backward()
            # 使用scaler进行梯度裁剪
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            # 使用scaler更新参数
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()
            train_mae += mae.item()

            # 实时更新进度条
            if batch_idx % 10 == 0:
                train_pbar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'MAE': f'{mae.item():.4f}',
                    'AvgLoss': f'{train_loss / (batch_idx + 1):.4f}'
                })

        avg_train_loss = train_loss / len(train_loader)
        avg_train_mae = train_mae / len(train_loader)

        # 验证阶段
        model.eval()
        val_loss = 0.0
        val_mae = 0.0
        val_mae_max = 0.
        val_pbar = tqdm(val_loader, desc=f'✅ 验证 Epoch {epoch + 1}/{num_epochs}',
                        bar_format='{l_bar}{bar:25}{r_bar}{bar:-10b}')

        with torch.no_grad():
            for batch_idx, (images, extra_features, labels) in enumerate(val_pbar):
                images, extra_features, labels = images.to(device), extra_features.to(device), labels.to(device)
                outputs = model(images, extra_features)
                loss = criterion(outputs, labels)
                ae = torch.abs(outputs - labels)
                mae = torch.mean(ae)
                val_mae_max = max(val_mae_max, ae.max())
                val_loss += loss.item()
                val_mae += mae.item()

                val_pbar.set_postfix({
                    'ValLoss': f'{loss.item():.4f}',
                    'ValMAE': f'{mae.item():.4f}',
                    'AvgValLoss': f'{val_loss / (batch_idx + 1):.4f}'
                })

        avg_val_loss = val_loss / len(val_loader)
        avg_val_mae = val_mae / len(val_loader)
        epoch_time = time.time() - epoch_start_time

        # 学习率调整
        scheduler.step(avg_val_loss)
        current_lr = optimizer.param_groups[0]['lr']

        # 记录历史
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['train_mae'].append(avg_train_mae)
        history['val_mae'].append(avg_val_mae)
        history['learning_rate'].append(current_lr)
        history['epoch_times'].append(epoch_time)

        # 早停检查
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_weights = model.state_dict().copy()
            torch.save(model.state_dict(), 'best_model.pth')
            patience_counter = 0
            improvement_msg = "✨ 新的最佳模型已保存!"
        else:
            patience_counter += 1
            improvement_msg = f"⏳ 早停计数: {patience_counter}/{early_stopping_patience}"

            if patience_counter >= early_stopping_patience:
                print(f"\n🚨 早停触发！在 Epoch {epoch + 1} 停止训练")
                break

        # 打印详细的epoch总结
        print(f"\n{'=' * 70}")
        print(f"📊 Epoch {epoch + 1}/{num_epochs} 总结:")
        print(f"  训练损失: {avg_train_loss:.6f} | 验证损失: {avg_val_loss:.6f}")
        print(f"  训练MAE:  {avg_train_mae:.6f} | 验证MAE:  {avg_val_mae:.6f}")
        print(f"  训练MAE_MAX:  {train_mae_max:.6f} | 验证MAE_MAX:  {val_mae_max:.6f}")
        print(f"  学习率: {current_lr:.3g} | 时间: {epoch_time:.2f}秒")
        print(f"  最佳验证损失: {best_val_loss:.6f}")
        print(f"  {improvement_msg}")
        print(f"{'=' * 70}\n")

    # 加载最佳模型
    if best_model_weights is not None:
        model.load_state_dict(best_model_weights)
        print(f"🎉 训练完成！加载最佳模型，验证损失: {best_val_loss:.6f}")
    return model, history


def parse_args():
    """
    解析命令行参数
    用法示例：
    python train.py --image_dirs img1 img2 --true_list 36.8 36.8 --batch_size 128 --epochs 500
    """
    parser = argparse.ArgumentParser(description='训练温度偏差校准模型')

    # 数据相关参数
    parser.add_argument('--image_dirs', nargs='+',
                        help='图片目录列表，用空格分隔多个目录')
    parser.add_argument('--true_list', nargs='+', type=float,
                        help='真实温度值列表，用空格分隔多个值')
    parser.add_argument('--train_ratio', type=float, default=0.8,
                        help='训练集比例，默认0.8')

    parser.add_argument('--model_dims', type=int, default=128,
                        help='模型隐藏层维度')

    # 训练相关参数
    parser.add_argument('--batch_size', type=int, default=128,
                        help='批次大小')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='数据加载工作进程数')
    parser.add_argument('--epochs', type=int, default=500,
                        help='训练轮数')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='学习率')
    parser.add_argument('--patience', type=int, default=10,
                        help='早停耐心值')

    return parser.parse_args()

# 主函数
def main():
    # 数据路径
    # 创建数据集
    args = parse_args()
    image_dir = args.image_dirs
    true_list = args.true_list
    batch_size = args.batch_size
    number_of_workers = args.num_workers
    model_dims = args.model_dims
    num_epochs = args.epochs
    lr = args.lr


    dataset = FaceData(image_dir,true_list)
    # 划分训练集和验证集
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=number_of_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=number_of_workers, pin_memory=True)

    # 创建模型
    model = CrossAttentionRegressionModel(hidden_dim=model_dims)
    # model = TransformerTemp(vector_in=8, vector_len=64, vector_dim=128, vector_layer=4, img_layer=4, decoder_layer=2)
    model.cuda()
    print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")
    print(f"训练样本数: {len(train_dataset)}")
    print(f"验证样本数: {len(val_dataset)}")

    # # 训练模型
    train_model(model, train_loader, val_loader, num_epochs=num_epochs, lr=lr,early_stopping_patience=10)
    print("模型已保存为 'temperature_regressor.pth'")

    # 测试预测
    model.load_state_dict(torch.load("best_model.pth", weights_only=True))
    model.eval()
    mse_list = []
    mae_list = []

    import pandas as pd
    with torch.no_grad():
        # 取一个批次进行测试
        test_images, test_extra, test_labels = next(iter(val_loader))
        test_images, test_extra = test_images.to('cuda' if torch.cuda.is_available() else 'cpu'), test_extra.to(
            'cuda' if torch.cuda.is_available() else 'cpu')
        predictions = model(test_images, test_extra).cpu()
        # 记录下来预测值和真实值的 mse 和 mae 我希望能在循环结束后 实例化存储，并且可视化
        mse_list.extend((predictions - test_labels).pow(2).tolist())
        mae_list.extend((abs(predictions - test_labels)).tolist())
    results = pd.DataFrame({'MSE': mse_list, 'MAE': mae_list})
    results.to_csv('results.csv', index=False)



if __name__ == "__main__":
    main()
