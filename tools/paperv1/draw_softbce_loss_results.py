import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline

# =========================
# 原始数据
# =========================
data = [
    [0.25, 90.69, 78.54, 72.91, 78.96],
    [0.50, 91.03, 78.98, 74.22, 79.83],
    [0.75, 90.23, 79.50, 75.18, 80.36],
    [0.80, 90.79, 81.62, 76.04, 81.28],
    [0.90, 92.25, 81.05, 74.65, 79.87],
    [1.00, 91.16, 82.41, 76.55, 81.14],
    [1.10, 89.88, 80.05, 74.65, 79.87],
    [1.20, 91.07, 81.20, 75.93, 80.58],
    [2.00, 91.23, 80.62, 75.30, 80.60],
    [4.00, 88.81, 78.31, 72.80, 78.32],
    [8.00, 89.27, 78.84, 72.30, 78.74],
]

# =========================
# 数据整理
# =========================
data = sorted(data, key=lambda x: x[0])

theta = np.array([x[0] for x in data])
map50 = np.array([x[1] for x in data])
map70 = np.array([x[2] for x in data])
map5090 = np.array([x[3] for x in data])
mar5090 = np.array([x[4] for x in data])

# =========================
# 分段平滑
# 第一段：0.25 ~ 1.2
# 第二段：1.2 ~ 8
# =========================
split_theta = 1.2
idx_left = theta <= split_theta
idx_right = theta >= split_theta

theta_left = theta[idx_left]
theta_right = theta[idx_right]

def segmented_smooth(x_left, y_left, x_right, y_right,
                     num_left=300, num_right=300,
                     k_left=3, k_right=3):
    """
    左段用二次样条平滑，右段用一次插值，避免大跨度导致曲线畸变
    """
    x_left_smooth = np.linspace(x_left.min(), x_left.max(), num_left)
    x_right_smooth = np.linspace(x_right.min(), x_right.max(), num_right)

    y_left_smooth = make_interp_spline(x_left, y_left, k=k_left)(x_left_smooth)
    y_right_smooth = make_interp_spline(x_right, y_right, k=k_right)(x_right_smooth)

    return x_left_smooth, y_left_smooth, x_right_smooth, y_right_smooth

map50_lx, map50_ly, map50_rx, map50_ry = segmented_smooth(
    theta_left, map50[idx_left], theta_right, map50[idx_right]
)
map70_lx, map70_ly, map70_rx, map70_ry = segmented_smooth(
    theta_left, map70[idx_left], theta_right, map70[idx_right]
)
map5090_lx, map5090_ly, map5090_rx, map5090_ry = segmented_smooth(
    theta_left, map5090[idx_left], theta_right, map5090[idx_right]
)
mar5090_lx, mar5090_ly, mar5090_rx, mar5090_ry = segmented_smooth(
    theta_left, mar5090[idx_left], theta_right, mar5090[idx_right]
)

# =========================
# 字体与全局参数
# =========================
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 12

fig, ax = plt.subplots(figsize=(8, 6))

# =========================
# 指定颜色
# =========================
c1 = '#1f77b4'   # mAP50
c2 = '#ff7f0e'   # mAP70
c3 = '#2ca02c'   # mAP50:90
c4 = '#d62728'   # mAR50:90

# =========================
# 绘制平滑曲线（左右两段同色）
# =========================
ax.plot(map50_lx, map50_ly, linewidth=2, color=c1, label=r'mAP$_{50}$')
ax.plot(map50_rx, map50_ry, linewidth=2, color=c1)

ax.plot(map70_lx, map70_ly, linewidth=2, color=c2, label=r'mAP$_{70}$')
ax.plot(map70_rx, map70_ry, linewidth=2, color=c2)

ax.plot(map5090_lx, map5090_ly, linewidth=2, color=c3, label=r'mAP$_{50:90}$')
ax.plot(map5090_rx, map5090_ry, linewidth=2, color=c3)

ax.plot(mar5090_lx, mar5090_ly, linewidth=2, color=c4, label=r'mAR$_{50:90}$')
ax.plot(mar5090_rx, mar5090_ry, linewidth=2, color=c4)

# =========================
# 绘制原始点（同色）
# =========================
ax.plot(theta, map50, 'o', markersize=5, color=c1)
ax.plot(theta, map70, 's', markersize=5, color=c2)
ax.plot(theta, map5090, '^', markersize=5, color=c3)
ax.plot(theta, mar5090, 'd', markersize=5, color=c4)

# =========================
# 标注 theta = 1.00
# =========================
ax.axvline(x=1.00, linestyle='--', linewidth=1.2, alpha=0.7, color='gray')

# =========================
# 坐标轴标签
# =========================
ax.set_xlabel(r'Hyperparameter $\theta$', fontsize=14)
ax.set_ylabel('Performance (%)', fontsize=14)

# 固定横轴范围，使 0 与左边界对齐
ax.set_xlim(0, 8.1)
ax.set_ylim(70, 92.5)

# 只显示少量横轴刻度
xticks_show = [0, 1.00, 2.00, 4.00, 8.00]
yticks_show = [70, 72.5, 75, 77.5, 80, 82.5, 85, 87.5, 90, 92.5]
ax.set_xticks(xticks_show)
ax.set_xticklabels([f'{x:.0f}' for x in xticks_show])
ax.set_yticks(yticks_show)
ax.set_yticklabels([f'{y:.1f}' for y in yticks_show])

# =========================
# 网格、图例、边框
# =========================
ax.grid(True, linestyle='--', alpha=0.4)
ax.legend(frameon=False, fontsize=12)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig('/mnt/d/paperv1/theta_ablation_curve.jpg', dpi=1200, bbox_inches='tight')
plt.show()