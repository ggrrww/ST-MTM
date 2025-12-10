import matplotlib.pyplot as plt
import numpy as np

# 原始的Loss数据
LOSS1 = [23.60,15.56,15.21,15.01,14.87,14.92,14.76,14.75,14.72,14.75,14.64,14.74,14.69,14.77,14.77,14.69,14.66,14.62,14.80,14.63,14.62]
LOSS2 = [20.06,14.89,14.44,14.25,14.22,14.20,14.04,13.89,13.79, 13.85, 13.96, 13.98, 13.83, 13.87, 13.78, 13.82, 13.92, 13.88, 14.05, 13.80, 13.90]

# 创建训练轮次数据
epochs = [1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200]
plt.xticks(epochs)
# 设置中文字体，避免中文显示问题
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号

# 创建图表
plt.figure(figsize=(10, 6))

# 绘制Loss曲线
plt.plot(epochs, LOSS1, 'b-o', linewidth=2, markersize=6, label='TMAE')
plt.plot(epochs, LOSS2, 'r-s', linewidth=2, markersize=6, label='PerioMAE')

# 添加网格线
plt.grid(True, linestyle='--', alpha=0.7)

# 添加标签和标题
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Loss', fontsize=12)
plt.title('Loss Compare', fontsize=14)

# 添加图例
plt.legend(fontsize=12)

# 设置坐标轴范围，使图表更美观
plt.ylim(min(min(LOSS1), min(LOSS2)) - 1, max(max(LOSS1), max(LOSS2)) + 1)
plt.xlim(0, 210)

# 添加数据点数值标注（可选）
# for i, value in enumerate(LOSS1):
#     if i % 2 == 0:  # 每隔一个点标注一次，避免过于密集
#         plt.annotate(f'{value:.2f}', xy=(epochs[i], value), xytext=(0, 10),
#                     textcoords='offset points', ha='center', fontsize=9, color='blue')

# for i, value in enumerate(LOSS2):
#     if i % 2 == 0:  # 每隔一个点标注一次，避免过于密集
#         plt.annotate(f'{value:.2f}', xy=(epochs[i], value), xytext=(0, -15),
#                     textcoords='offset points', ha='center', fontsize=9, color='red')

# 计算并显示最终Loss值
plt.axhline(y=LOSS1[-1], color='blue', linestyle='-.', alpha=0.5)
plt.axhline(y=LOSS2[-1], color='red', linestyle='-.', alpha=0.5)
plt.text(len(epochs) + 0.5, LOSS1[-1], f'Loss1: {LOSS1[-1]:.2f}', color='blue', fontsize=10)
plt.text(len(epochs) + 0.5, LOSS2[-1], f'Loss2: {LOSS2[-1]:.2f}', color='red', fontsize=10)

# 保存图表（可选）
plt.savefig('stdmae/fig/loss_comparison.png', dpi=300, bbox_inches='tight')