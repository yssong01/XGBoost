import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.datasets import make_circles
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from mpl_toolkits.mplot3d import Axes3D
import os

# ==========================================
# 1. 데이터 생성 (High Noise)
# ==========================================
np.random.seed(42)

# [A] 회귀 (Regression) - 노이즈가 매우 심한 데이터(가정)
X_reg = np.random.uniform(0, 5, (200, 1))
noise = np.random.normal(0, 1.5, X_reg.shape) 
y_reg = (X_reg * np.sin(X_reg)).ravel() + noise.ravel()
X_test_reg = np.arange(0.0, 5.0, 0.01)[:, np.newaxis]

# [B] 분류 (Classification)
X_cls, y_cls = make_circles(n_samples=300, factor=0.5, noise=0.2, random_state=42)

# 저장 폴더
output_dir = 'dashboard_GB_imgs'
os.makedirs(output_dir, exist_ok=True)

# ==========================================
# 2. Stage & 3D Simulation Config
# ==========================================
# 전체 시뮬레이션
range1 = np.arange(1, 21, 1)
range2 = np.arange(21, 101, 1)
range3 = np.arange(101, 301, 1)
stages = np.unique(np.concatenate((range1, range2, range3)))


# 3D 목표 지점
TARGET_X = -4.24
TARGET_Y = -3.42
MIN_EXPECTED_LOSS = 2.0 

# 3D 궤적 시뮬레이션
t = np.linspace(0, 12, len(stages))
opt_x = TARGET_X + 6.0 * np.exp(-0.25 * t) * np.cos(1.5 * t + 1)
opt_y = TARGET_Y + 6.0 * np.exp(-0.25 * t) * np.sin(1.5 * t + 1)

csv_data = []

# ==========================================
# 3. 스타일 설정
# ==========================================
FS_TITLE = 20
FS_LABEL = 15
FS_TICK = 12
FW = 'bold'

# ==========================================
# 4. Main Loop
# ==========================================
print(f"Total stages: {len(stages)}")

for idx, s in enumerate(stages):
    # --- 모델 학습 ---
    reg_model = GradientBoostingRegressor(n_estimators=s, learning_rate=0.1, max_depth=3, random_state=42)
    reg_model.fit(X_reg, y_reg)
    y_reg_pred = reg_model.predict(X_reg)
    y_reg_test = reg_model.predict(X_test_reg)
    
    cls_model = GradientBoostingClassifier(n_estimators=s, learning_rate=0.1, max_depth=3, random_state=42)
    cls_model.fit(X_cls, y_cls)

    # --- Data Logging ---
    loss = reg_model.train_score_[-1]
    
    cur_x, cur_y = opt_x[idx], opt_y[idx]
    
    # [보정] 궤적이 곡면 위에 붙도록 Z값 보정 (시각적용)
    visual_z = 0.05 * ((cur_x - TARGET_X)**2 + (cur_y - TARGET_Y)**2) + MIN_EXPECTED_LOSS
    
    csv_data.append({
        'Stage': s, 
        'Learning Rate (Hyperparameter, X)': cur_x, 
        'Max Depth (Hyperparameter, Y)': cur_y, 
        'Total Loss | Cost (MSE)': visual_z
    })

    # --- 캔버스 생성 ---
    fig = plt.figure(figsize=(20, 16))

    # -------------------------------------------------------
    # [Q2] Decision Boundary (Top-Left)
    # -------------------------------------------------------
    ax2 = fig.add_subplot(2, 2, 1)
    x_min, x_max = X_cls[:, 0].min() - 0.5, X_cls[:, 0].max() + 0.5
    y_min, y_max = X_cls[:, 1].min() - 0.5, X_cls[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02), np.arange(y_min, y_max, 0.02))
    Z_cls = cls_model.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1].reshape(xx.shape)
    
    levels = np.linspace(0, 1.0, 21)
    cf = ax2.contourf(xx, yy, Z_cls, levels=levels, cmap='coolwarm', alpha=0.8)
    ax2.scatter(X_cls[:, 0], X_cls[:, 1], c=y_cls, cmap='coolwarm', edgecolors='k', s=40)
    
    ax2.set_title(f'[Q2] Decision Boundary (Stage {s})', fontsize=FS_TITLE, fontweight=FW)
    ax2.set_xlabel('Alcohol Content (%)', fontsize=FS_LABEL, fontweight=FW)
    ax2.set_ylabel('Malic Acid (%)', fontsize=FS_LABEL, fontweight=FW)
    cbar2 = plt.colorbar(cf, ax=ax2, ticks=np.arange(0, 1.1, 0.1))
    cbar2.set_label('Probability (Confidence)', fontsize=FS_LABEL, fontweight=FW)

    # -------------------------------------------------------
    # [Q1] Function Fitting (Top-Right)
    # -------------------------------------------------------
    ax1 = fig.add_subplot(2, 2, 2)
    ax1.scatter(X_reg, y_reg, color='gray', s=30, alpha=0.6, label='Noisy Input')
    ax1.plot(X_test_reg, y_reg_test, color='red', linewidth=2, label='Model Pred')
    
    ax1.set_title(f'[Q1] Function Fitting (Stage {s})', fontsize=FS_TITLE, fontweight=FW)
    ax1.set_xlabel('Signal Intensity (V)', fontsize=FS_LABEL, fontweight=FW)
    ax1.set_ylabel('Response Current (mA)', fontsize=FS_LABEL, fontweight=FW)
    ax1.legend(loc='upper right', fontsize=FS_LABEL, markerscale=3, borderpad=1.5, framealpha=0.9, shadow=True)
    ax1.grid(True, linestyle='--', alpha=0.5)

    # -------------------------------------------------------
    # [Q3] Training Loss + Inset (Bottom-Left)
    # -------------------------------------------------------
    ax3 = fig.add_subplot(2, 2, 3)
    loss_hist = reg_model.train_score_
    iters = np.arange(1, len(loss_hist) + 1)
    
    ax3.plot(iters, loss_hist, color='blue', linewidth=3, marker='o', markersize=6)
    ax3.scatter(iters[-1], loss_hist[-1], color='red', s=150, zorder=5)

    # --- [수정 핵심] 가로축 범위를 0부터 300(또는 n_samples)으로 고정 ---
    ax3.set_xlim(0, 300)
    
    # [반영 1] 사용자 요청 좌표 조정 (iters[-1]-0.1, loss_hist[-1]+0.05)
    ax3.text(iters[-1], loss_hist[-1]+0.05, f'  {loss_hist[-1]:.2f}', 
             color='red', fontsize=FS_LABEL, fontweight=FW,
             verticalalignment='bottom', horizontalalignment='left')
    
    ax3.set_title(f'[Q3] Training Loss (Stage {s})', fontsize=FS_TITLE, fontweight=FW)
    ax3.set_xlabel('Boosting Iterations', fontsize=FS_LABEL, fontweight=FW)
    ax3.set_ylabel('Loss (MSE)', fontsize=FS_LABEL, fontweight=FW)
    ax3.grid(True, alpha=0.5)

    # --- Inset: Actual vs Predicted ---
    axins = inset_axes(ax3, width="100%", height="100%", 
                       bbox_to_anchor=(0.3, 0.25, 0.65, 0.65), 
                       bbox_transform=ax3.transAxes,
                       loc='upper right', borderpad=0)
    
    errs = np.abs(y_reg - y_reg_pred)
    sc = axins.scatter(y_reg, y_reg_pred, c=errs, cmap='viridis_r', s=30, edgecolors='k', alpha=0.8)
    
    min_v, max_v = min(y_reg.min(), y_reg_pred.min()), max(y_reg.max(), y_reg_pred.max())
    axins.plot([min_v, max_v], [min_v, max_v], 'r--', linewidth=2)
    
    axins.set_aspect('equal', adjustable='box')
    
    for spine in axins.spines.values():
        spine.set_edgecolor('green')
        spine.set_linewidth(3)
    
    axins.set_title('[Inset] Actual vs Pred', fontsize=15, fontweight=FW)
    axins.set_xlabel('Actual', fontsize=15, fontweight='bold')
    axins.set_ylabel('Pred', fontsize=15, fontweight='bold')
    axins.tick_params(axis='both', which='major', labelsize=15)
    axins.grid(True, linestyle='--', alpha=0.5, color='gray')
    
    cax_ins = inset_axes(axins, width="5%", height="100%", loc='lower left',
                         bbox_to_anchor=(1.05, 0., 1, 1), bbox_transform=axins.transAxes, borderpad=0)
    cbar_ins = plt.colorbar(sc, cax=cax_ins)
    cbar_ins.set_label('Error', fontsize=15, fontweight='bold')
    cbar_ins.ax.tick_params(labelsize=8)

    # -------------------------------------------------------
    # [Q4] 3D Optimization (Bottom-Right)
    # -------------------------------------------------------
    ax4 = fig.add_subplot(2, 2, 4, projection='3d')
    
    # Surface Landscape
    X_surf = np.linspace(-10, 2, 30)
    Y_surf = np.linspace(-10, 2, 30)
    X_surf, Y_surf = np.meshgrid(X_surf, Y_surf)
    # 배경 곡면 Z (visual_z와 동일 수식)
    Z_surf = 0.05 * ((X_surf - TARGET_X)**2 + (Y_surf - TARGET_Y)**2) + MIN_EXPECTED_LOSS
    
    surf = ax4.plot_surface(X_surf, Y_surf, Z_surf, cmap='rainbow', alpha=0.3, 
                            edgecolor='gray', linewidth=0.5, rstride=1, cstride=1)
    
    # Trajectory
    hist_x = [d['Learning Rate (Hyperparameter, X)'] for d in csv_data]
    hist_y = [d['Max Depth (Hyperparameter, Y)'] for d in csv_data]
    hist_z = [d['Total Loss | Cost (MSE)'] for d in csv_data]
    
    ax4.plot(hist_x, hist_y, hist_z, color='black', linewidth=1.5, linestyle='--')
    ax4.scatter(cur_x, cur_y, visual_z, color='red', s=250, marker='*', zorder=10, edgecolors='white')
    
    # Stage # 라벨 (별 따라다님)
    ax4.text(cur_x, cur_y, visual_z + 1.0, f"Stage {s}", color='red', fontsize=14, fontweight='bold')

    # [반영 2] 상세 정보 표 위치 조정 (0.9, 0.8) -> 우측 상단
    fixed_label_text = f"BEST POINT\nX: {cur_x:.2f}\nY: {cur_y:.2f}\nZ(Loss): {visual_z:.2f}"
    bbox_props = dict(boxstyle="round,pad=0.5", fc="white", ec="red", lw=2, alpha=0.9)
    
    ax4.text2D(0.9, 0.8, fixed_label_text, transform=ax4.transAxes,
               color='red', fontsize=13, fontweight='bold',
               ha='right', va='bottom', bbox=bbox_props)

    ax4.set_title(f'[Q4] Loss Landscape Optimization', fontsize=FS_TITLE, fontweight=FW)
    ax4.set_xlabel('Learning Rate (Hyperparameter, X)', fontsize=FS_LABEL, fontweight=FW)
    ax4.set_ylabel('Max Depth (Hyperparameter, Y)', fontsize=FS_LABEL, fontweight=FW)
    ax4.set_zlabel('Total Loss | Cost (MSE)', fontsize=FS_LABEL, fontweight=FW)


    # [수정 5] Z축 범위 설정 (0 ~ 곡면 최대 높이 * 1.2)
    ax4.set_zlim(0, Z_surf.max() * 1.1)
    
    ax4.view_init(elev=35, azim=45 + idx/3)
    cbar4 = fig.colorbar(surf, ax=ax4, shrink=0.6, pad=0.1)
    cbar4.set_label('Optimization Cost (Loss)', fontsize=FS_LABEL, fontweight=FW)

    # --- [수정 핵심] tight_layout 경고 해결 ---
    # plt.tight_layout() # <-- 경고 발생 원인
    
    # 대신 수동으로 여백 조정 (2행 2열 구조에 맞춰 넉넉하게 설정)
    # left, right, top, bottom: 전체 그림의 여백
    # wspace, hspace: 서브플롯 간의 가로, 세로 간격
    plt.subplots_adjust(left=0.05, right=0.95, top=0.92, bottom=0.05, wspace=0.15, hspace=0.2)
    
    file_path = os.path.join(output_dir, f'dashboard_stage_{s:03d}.png')
    plt.savefig(file_path)
    plt.close()
    
    if idx % 10 == 0:
        print(f"Processed Stage {s} (GB)...")

# CSV 저장
df_log = pd.DataFrame(csv_data)
df_log.to_csv('optimization_log_GB.csv', index=False)
print("!! Visualization & CSV Export Complete!")