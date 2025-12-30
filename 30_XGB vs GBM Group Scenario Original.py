import pandas as pd
import numpy as np
import os
import time
import gc
import warnings

# Matplotlib의 "백엔드(Backend)"를 설정하는 명령.--------------
# "화면에 그래프 창을 띄우지 않고, 메모리상에서 그림을 그려 파일(PNG, PDF 등)로 저장만 한다."**
import matplotlib
matplotlib.use('Agg') 
#----------------------------------------------------------

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import MaxNLocator

# XGBoost
import xgboost as xgb
# GBM (Scikit-learn)
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier

from sklearn.metrics import mean_squared_error, r2_score
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from matplotlib.colors import ListedColormap

warnings.filterwarnings("ignore")


# ==========================================================
# [PROJECT MASTER SETTING]
# ==========================================================

DATA_PATH = './wine_raw_data_initial.csv' 

SCENARIOS = {
    1:  ('G1_Orthogonality', 'Alcohol', 'Flavanoids'),
    2:  ('G1_Orthogonality', 'Flavanoids', 'Malic Acid'),
    3:  ('G2_Redundancy', 'Flavanoids', 'Total Phenols'),
    4:  ('G2_Redundancy', 'Flavanoids', 'Od280/Od315 Of Diluted Wines'),
    5:  ('G3_Scale', 'Proline', 'Flavanoids'),
    6:  ('G3_Scale', 'Proline', 'Color Intensity'),
    7:  ('G4_Noise', 'Ash', 'Magnesium'),
    8:  ('G4_Noise', 'Color Intensity', 'Hue'),
    9:  ('G5_Physics', 'Alcohol', 'Proline'),
    10: ('G5_Physics', 'Total Phenols', 'Hue')
}

F_LAB = {'fontsize': 25, 'fontweight': 'bold'}
TICK_SIZE_MAIN = 20 
TICK_SIZE_Q4 = 20 
TITLE_PAD = 35 

# ----------------------------------------------------------
# 데이터 로드
# ----------------------------------------------------------
df_main = pd.read_csv(DATA_PATH)
# CSV 저장 시 'Label'로 저장했으므로 대문자로 변경
y_cls_label = df_main['Label'].values 
wine_colors = ['#FF8C00', '#228B22', '#0000FF']
wine_cmap = ListedColormap(wine_colors[:len(np.unique(y_cls_label))])

# ==========================================================
# 시나리오 전체 순회
# ==========================================================
# 1~99까지 step 1, 100~1000까지 step 5 리스트 생성
#----------------------------------------------------------
stage_range_1 = range(1, 19+1, 1)    # 1~19까지 step 1
stage_range_2 = range(20, 1000+1, 5)  # 20~1000까지 step 5

stages_to_run = list(stage_range_1) + list(stage_range_2)

#for SCENARIO_SELECTOR in (4,):
for SCENARIO_SELECTOR in (1,3,5,7,9,2,4,6,8,10): # 시나리오 선택
#for SCENARIO_SELECTOR in (1,3,5,7,9): # 시나리오 선택
#for SCENARIO_SELECTOR in (2,4,6,8,10): # 시나리오 선택
#----------------------------------------------------------
    
    GROUP_NAME, X_COL, Y_COL = SCENARIOS[SCENARIO_SELECTOR]
    # 폴더명 기본 베이스 (X_COL의 '/' 문자 치환 포함)

    # [수정 전]
    # base_scenario_name = f'{GROUP_NAME}_Scenario{SCENARIO_SELECTOR}_{X_COL.replace("/", "_")}'

    # [수정 후] 폴더명에 'Original' 명시
    base_scenario_name = f'{GROUP_NAME}_Scenario{SCENARIO_SELECTOR}_{X_COL.replace("/", "_")}_Original'

    print(f"\n>>> [START] Scenario {SCENARIO_SELECTOR}: {GROUP_NAME} ({X_COL} vs {Y_COL})")

    # 데이터 전처리
    df = df_main.copy()
    X_data, Y_data = df[[X_COL]].values, df[[Y_COL]].values
    
    df['x_bin'] = pd.cut(df[X_COL], bins=3, labels=[0, 1, 2]).astype(int)
    df['y_bin'] = pd.cut(df[Y_COL], bins=3, labels=[0, 1, 2]).astype(int)
    df['region_id'] = df['x_bin'] * 3 + df['y_bin']
    R_id = df[['region_id']].values

    x_min, x_max = X_data.min(), X_data.max()
    y_min, y_max = Y_data.min(), Y_data.max()
    x_pad, y_pad = (x_max - x_min)*0.15, (y_max - y_min)*0.15
    xx, yy = np.meshgrid(np.linspace(x_min-x_pad, x_max+x_pad, 100), np.linspace(y_min-y_pad, y_max+y_pad, 100))

    x_grid_lines = np.linspace(x_min, x_max, 4)
    y_grid_lines = np.linspace(y_min, y_max, 4)

    # 기록용 리스트
    loss_xgb, time_xgb, param_history_xgb = [], [], []
    loss_gbm, time_gbm, param_history_gbm = [], [], []

    # ==========================================================
    # 시뮬레이션 루프 (Stages)
    # ==========================================================
    for s in stages_to_run:
        # 9구역 모델과 동일한 하이퍼파라미터 스케줄을 적용합니다.
        cur_lr = max(0.005, 0.3 * (0.995 ** s)) 
        cur_depth = min(15, int(4 + s / 100))

        # 10 스테이지마다 한 번씩만 진행 상황 보고
        # [수정] 현재 어떤 시나리오의 어떤 스테이지인지 터미널에 명확히 출력
        print(f"    -> [Scenario {SCENARIO_SELECTOR}/10] {GROUP_NAME} | Processing Stage: {s} ...          ", end='\r')

        # [수정 전]
        # cur_lr = max(0.005, 0.3 * (0.995 ** s)) 
        # cur_depth = min(15, int(4 + s / 100))
        # [수정 후] Learning Rate 및 Max Depth 스케줄링 조정
        # 표준 Baseline 수치로 고정합니다.
        # cur_lr = 0.05  
        # cur_depth = 6
        
        # ------------------------------------------------------
        # 1. XGBoost 학습
        # ------------------------------------------------------
        st_time = time.time()
        # m_a_xgb = xgb.XGBRegressor(n_estimators=s, learning_rate=cur_lr, max_depth=cur_depth, random_state=42, n_jobs=-1)
        # XGBRegressor (m_a_xgb, m_b_xgb) 부분
        m_a_xgb = xgb.XGBRegressor(
            n_estimators=s, 
            learning_rate=cur_lr, 
            max_depth=cur_depth, 
            random_state=42, 
            tree_method='hist',   # GPU 사용을 위한 히스토그램 기반 알고리즘
            device='cuda'         # 연산 장치를 GPU(CUDA)로 지정
        )


        # [수정 전]
        # m_a_xgb.fit(np.column_stack((X_data, R_id)), Y_data)
        # y_p_a_xgb = m_a_xgb.predict(np.column_stack((X_data, R_id)))

        # [수정 후] R_id 제거
        m_a_xgb.fit(X_data, Y_data)
        y_p_a_xgb = m_a_xgb.predict(X_data)
        
        m_b_xgb = xgb.XGBRegressor(n_estimators=s, learning_rate=cur_lr, max_depth=cur_depth, random_state=42, n_jobs=-1)

        # [수정 전]
        # m_b_xgb.fit(np.column_stack((Y_data, R_id)), X_data)
        # x_p_b_xgb = m_b_xgb.predict(np.column_stack((Y_data, R_id)))

        # m_b_xgb 부분도 마찬가지로 (Y_data, R_id) -> Y_data로 수정
        m_b_xgb.fit(Y_data, X_data)
        x_p_b_xgb = m_b_xgb.predict(Y_data)


        time_xgb.append(time.time() - st_time)
        
        cur_loss_xgb = (mean_squared_error(Y_data, y_p_a_xgb) + mean_squared_error(X_data, x_p_b_xgb)) / 2
        loss_xgb.append(cur_loss_xgb)

        # R2 신뢰도 계산
        r2_y_xgb = r2_score(Y_data, y_p_a_xgb)
        r2_x_xgb = r2_score(X_data, x_p_b_xgb)

        # m_cls_xgb = xgb.XGBClassifier(n_estimators=s, learning_rate=cur_lr, max_depth=cur_depth, eval_metric='logloss', random_state=42, n_jobs=-1)
        # XGBClassifier (m_cls_xgb) 부분
        m_cls_xgb = xgb.XGBClassifier(
            n_estimators=s, 
            learning_rate=cur_lr, 
            max_depth=cur_depth, 
            eval_metric='logloss', 
            random_state=42, 
            tree_method='hist', 
            device='cuda'
        )


        m_cls_xgb.fit(np.column_stack((X_data, Y_data)), y_cls_label)
        Z_cls_xgb = m_cls_xgb.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1].reshape(xx.shape)
        
        # CSV 저장을 위해 R2_X, R2_Y 추가
        param_history_xgb.append([s, cur_lr, cur_depth, cur_loss_xgb, r2_x_xgb, r2_y_xgb])

        # ------------------------------------------------------
        # 2. GBM 학습
        # ------------------------------------------------------
        st_time_g = time.time()
        g_a_gbm = GradientBoostingRegressor(n_estimators=s, learning_rate=cur_lr, max_depth=cur_depth, random_state=42)

        # [수정 전]
        # g_a_gbm.fit(np.column_stack((X_data, R_id)), Y_data.ravel())
        # y_p_a_gbm = g_a_gbm.predict(np.column_stack((X_data, R_id))).reshape(-1, 1)

        # [수정 후] R_id 제거
        g_a_gbm.fit(X_data, Y_data.ravel())
        y_p_a_gbm = g_a_gbm.predict(X_data).reshape(-1, 1)


        g_b_gbm = GradientBoostingRegressor(n_estimators=s, learning_rate=cur_lr, max_depth=cur_depth, random_state=42)

        # [수정 전]
        # g_b_gbm.fit(np.column_stack((Y_data, R_id)), X_data.ravel())
        # x_p_b_gbm = g_b_gbm.predict(np.column_stack((Y_data, R_id))).reshape(-1, 1)

        # [수정 후] R_id 제거
        g_b_gbm.fit(Y_data, X_data.ravel())
        x_p_b_gbm = g_b_gbm.predict(Y_data).reshape(-1, 1)


        time_gbm.append(time.time() - st_time_g)

        cur_loss_gbm = (mean_squared_error(Y_data, y_p_a_gbm) + mean_squared_error(X_data, x_p_b_gbm)) / 2
        loss_gbm.append(cur_loss_gbm)

        # R2 신뢰도 계산
        r2_y_gbm = r2_score(Y_data, y_p_a_gbm)
        r2_x_gbm = r2_score(X_data, x_p_b_gbm)

        g_cls_gbm = GradientBoostingClassifier(n_estimators=s, learning_rate=cur_lr, max_depth=cur_depth, random_state=42)
        g_cls_gbm.fit(np.column_stack((X_data, Y_data)), y_cls_label)
        Z_cls_gbm = g_cls_gbm.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1].reshape(xx.shape)

        param_history_gbm.append([s, cur_lr, cur_depth, cur_loss_gbm, r2_x_gbm, r2_y_gbm])

        # ------------------------------------------------------
        # 3. 모델별 데이터 저장 및 시각화 (개선된 루프)
        # ------------------------------------------------------
        models_data = {
            'XGB': {
                'loss_hist': loss_xgb, 'time_hist': time_xgb, 'param_hist': param_history_xgb,
                'Z_cls': Z_cls_xgb, 'y_pred_a': y_p_a_xgb, 'x_pred_b': x_p_b_xgb,
                'model_a': m_a_xgb, 'model_b': m_b_xgb,
                'r2_vals': [r2_x_xgb, r2_y_xgb] # 추가
            },
            'GBM': {
                'loss_hist': loss_gbm, 'time_hist': time_gbm, 'param_hist': param_history_gbm,
                'Z_cls': Z_cls_gbm, 'y_pred_a': y_p_a_gbm, 'x_pred_b': x_p_b_gbm,
                'model_a': g_a_gbm, 'model_b': g_b_gbm,
                'r2_vals': [r2_x_gbm, r2_y_gbm] # 추가
            }
        }

        for m_name, m_dat in models_data.items():
            # 전용 폴더 생성 (예: G1_Orthogonality_Scenario1_XGB)
            specific_folder = f"{base_scenario_name}_{m_name}"
            os.makedirs(specific_folder, exist_ok=True)
            
            # 파일명 베이스 생성 (예: G1_Orthogonality_Scenario1_XGB_stage_1)
            # [수정 전]
            # file_base_name = f"{base_scenario_name}_{m_name}_stage_{s}"

            # [수정 후] 'Baseline' 또는 'Original' 추가
            file_base_name = f"{base_scenario_name}_{m_name}_Original_stage_{s}"

            
            # (1) CSV 저장
            df_log = pd.DataFrame(m_dat['param_hist'], columns=['Stage', 'Learning_Rate', 'Max_Depth', 'MSE_Loss', 'R2_X', 'R2_Y'])
            df_log['Avg_Time'] = np.mean(m_dat['time_hist'])
            df_log.to_csv(os.path.join(specific_folder, f"{file_base_name}.csv"), index=False)

            # (2) 시각화 데이터 준비
            _loss = m_dat['loss_hist']
            _time = m_dat['time_hist']
            _param = np.array(m_dat['param_hist']) # [Stage, LR, Depth, Loss]
            _Z = m_dat['Z_cls']
            _ypa = m_dat['y_pred_a']
            _xpb = m_dat['x_pred_b']
            _ma = m_dat['model_a']
            _mb = m_dat['model_b']
            _r2s = m_dat['r2_vals']

            # [권장] figsize를 22 정도로 낮추면 훨씬 안정적으로 저장
            fig = plt.figure(figsize=(22, 22)) 
            fig.suptitle(f"{GROUP_NAME} Scenario {SCENARIO_SELECTOR} | {m_name} | Stage {s}", fontsize=45, fontweight='bold', color='navy')

            # [Q1] Boundary Map
            ax1 = fig.add_subplot(2, 2, 1)
            ax1.set_box_aspect(1)
            cont = ax1.contourf(xx, yy, _Z, levels=np.arange(0, 1.05, 0.05), cmap='RdYlBu_r', alpha=0.5)
            for i, name in enumerate(['wine A', 'wine B', 'wine C']):
                mask = (y_cls_label == i)
                if mask.any(): ax1.scatter(X_data[mask], Y_data[mask], c=wine_colors[i], label=name, edgecolors='black', s=120, zorder=5)
            # -----------------------------------------------------------
            # [삭제/주석 처리] 9구역을 나누는 이 2줄을 지워주세요.
            # -----------------------------------------------------------
            # for l in x_grid_lines: ax1.axvline(l, color='black', lw=2, ls='--', alpha=0.4, zorder=1)
            # for l in y_grid_lines: ax1.axhline(l, color='black', lw=2, ls='--', alpha=0.4, zorder=1)
            # -----------------------------------------------------------
            ax1.set_xlabel(X_COL + ' (X)', fontsize=25, fontweight='bold', color='blue', labelpad=20)
            ax1.set_ylabel(Y_COL + ' (Y)', fontsize=25, fontweight='bold', color='blue', labelpad=20)
            ax1.tick_params(axis='both', labelsize=TICK_SIZE_MAIN)
            cax1 = ax1.inset_axes([1.05, 0, 0.05, 1])
            cbar = fig.colorbar(cont, cax=cax1)
            cbar.set_label('Prediction Confidence', fontsize=20, fontweight='bold', labelpad=15)
            cbar.ax.tick_params(labelsize=15)
            ax1.set_title(f'[1] Decision Topology Map ({s})', fontsize=30, fontweight='bold', pad=TITLE_PAD)
            ax1.legend(fontsize=18, loc='upper right')

            # [Q2] Experts
            # [Q2] Global Regression Analysis (Original 모델용으로 단순화)
            ax2 = fig.add_subplot(2, 2, 2)
            ax2.set_box_aspect(1)
            
            # 축 범위 고정
            ax2.set_xlim(x_min - x_pad, x_max + x_pad)
            ax2.set_ylim(y_min - y_pad, y_max + y_pad)

            # 데이터 산점도 출력
            for i in range(len(wine_colors)):
                mask = (y_cls_label == i)
                if mask.any():
                    ax2.scatter(X_data[mask], Y_data[mask], c=wine_colors[i], s=120, edgecolors='black', alpha=0.6, zorder=2)
            
            # -----------------------------------------------------------
            # [삭제/주석 처리] 여기서도 이 2줄을 지워주세요.
            # -----------------------------------------------------------
            # for l in x_grid_lines: ax2.axvline(l, color='black', lw=2, ls='--', alpha=0.4, zorder=1)
            # for l in y_grid_lines: ax2.axhline(l, color='black', lw=2, ls='--', alpha=0.4, zorder=1)
            # -----------------------------------------------------------

            xr, yr = np.linspace(x_min, x_max, 100).reshape(-1, 1), np.linspace(y_min, y_max, 100).reshape(-1, 1)

            # -----------------------------------------------------------
            # [핵심 수정] Baseline은 영역 구분이 없으므로 r 루프를 삭제하고 
            # 1개의 피처(xr 또는 yr)만 단독으로 넣어 예측합니다.
            # -----------------------------------------------------------
            ax2.plot(xr, _ma.predict(xr), color='black', lw=5, label='Global Prediction (X)', zorder=10)
            ax2.plot(_mb.predict(yr), yr, color='red', lw=5, label='Global Prediction (Y)', zorder=10)
            # -----------------------------------------------------------

            ax2.set_xlabel(X_COL + ' (X)', fontsize=25, fontweight='bold', color='blue', labelpad=20)
            ax2.set_ylabel(Y_COL + ' (Y)', fontsize=25, fontweight='bold', color='blue', labelpad=20)
            ax2.tick_params(axis='both', labelsize=TICK_SIZE_MAIN)
            
            ax2.legend(fontsize=16, loc='upper right', framealpha=0.8) 
            ax2.set_title(f'[2] Global Regression Analysis ({s})', fontsize=30, fontweight='bold', pad=TITLE_PAD)


            # [Q3] Convergence
            ax3 = fig.add_subplot(2, 2, 3)
            ax3.set_box_aspect(1)

            # (A) 오프셋 계산을 위한 Loss 범위 산출
            y_min_loss = np.min(_loss)
            y_max_loss = np.max(_loss)
            loss_range = y_max_loss - y_min_loss if y_max_loss != y_min_loss else 0.1

            # (B) 그래프 선 및 마지막 점 출력
            # 1. 실제 Stage 값(_param[:, 0])을 X축으로 하여 선을 하나만 그립니다.
            ax3.plot(_param[:, 0], _loss, 'bo-', label=f'{m_name} Loss', lw=2.5)
            
            # 2. 마지막 점 강조 (실제 Stage 위치: _param[-1, 0])
            ax3.scatter(_param[-1, 0], _loss[-1], color='red', marker='o', s=250, zorder=15)
            
            # 3. 마지막 수치 텍스트 표시 (실제 Stage 위치 기준)
            # -----------------------------------------------------------
            # [핵심 수정] 텍스트 높이를 (마지막값 + 3% Range)로 조정
            # -----------------------------------------------------------
            label_y_pos = _loss[-1] + 0.03 * loss_range
            ax3.text(_param[-1, 0], label_y_pos, f'{_loss[-1]:.4f}', 
                     color='red', fontweight='bold', fontsize=26, ha='center', va='bottom')
            # -----------------------------------------------------------
            
            # 2. 라벨 수정 (STAGE_STEP 반영 취소 및 원상 복구)
            ax3.set_xlabel('Training Stage', fontsize=25, fontweight='bold', color='black', labelpad=20)
            ax3.set_ylabel('MSE Loss', fontsize=25, fontweight='bold', color='black', labelpad=20)

            ax3.tick_params(axis='both', labelsize=TICK_SIZE_MAIN)
            ax3.legend(fontsize=18)
            ax3.grid(True, alpha=0.2, linewidth=2.0)
            ax3.set_title(f'[3] Model Convergence Analysis ({s})', fontsize=30, fontweight='bold', pad=TITLE_PAD)
            ax3.text(0.45, 0.22, f'Avg Time ({m_name}): {np.mean(_time):.4f}s', transform=ax3.transAxes, fontsize=20, fontweight='bold', bbox=dict(facecolor='white', alpha=0.8))
            
            # Inset 그래프 루프 수정
            ins_s = 0.42 
            # 각 inset별 r2값을 매칭하기 위해 zip 사용
            inset_configs = [
                (X_data, _xpb, f'{X_COL}', [0.22, 0.40, ins_s, ins_s], _r2s[0]),
                (Y_data, _ypa, f'{Y_COL}', [0.78, 0.40, ins_s, ins_s], _r2s[1])
            ]
            
            for d, p, t, pos, r2_val in inset_configs:
                ins = ax3.inset_axes(pos)
                ins.set_box_aspect(1)
                ins.tick_params(axis='both', labelsize=20, width=2)
                ins.scatter(d, p, c=y_cls_label, cmap=wine_cmap, s=40, alpha=0.6, edgecolors='black')
                ins.plot([d.min(), d.max()], [d.min(), d.max()], 'r--', lw=2.5)
                ins.grid(True, color='gray', linestyle='--', alpha=0.3, linewidth=2.0)
                ins.set_xlabel('Actual', fontsize=25, fontweight='bold')
                ins.set_ylabel('Pred.', fontsize=25, fontweight='bold')
                ins.set_title(t, fontsize=25, fontweight='bold', color='blue') # 소문자 유지 (변수 t 자체가 소문자임)
                # Inset 내부에 R2 텍스트 추가
                ins.text(0.95, 0.05, f'$R^2$: {r2_val:.3f}', transform=ins.transAxes, 
                         fontsize=25, fontweight='bold', ha='right', va='bottom', 
                         bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))


            # [Q4] 3D Landscape 통합 및 최적화 버전
            # [Q4] Loss Projection Landscape (GPR 제외 및 데이터 궤적/투영 강화)
            # [Q4] 하이브리드 시각화: Z축(선형) + 바닥(로그 컨투어) + 중력 우물
            ax4 = fig.add_subplot(2, 2, 4, projection='3d')
            ax4.set_box_aspect((1, 1, 1), zoom=1.1)
            ax4.computed_zorder = False 

            if _param.shape[0] > 0:
                # 1. 데이터 기초 정보 및 고정 범위 설정
                z_raw_val = _param[:, 3]
                x_raw_val, y_raw_val = _param[:, 1], _param[:, 2]
                cx, cy, cz = _param[-1, 1], _param[-1, 2], _param[-1, 3]
                
                # [수정] 박사님께서 요청하신 표준 범위 설정
                xr_3d, yr_3d = [0.0, 0.35], [0.0, 16.0] 
                x_len = xr_3d[1] - xr_3d[0] # 0.35
                y_len = yr_3d[1] - yr_3d[0] # 16.0

                z_max_global_val = np.max(z_raw_val)
                ax4.set_zlim(0, z_max_global_val * 1.2)
                ax4.set_xlim(xr_3d[0], xr_3d[1])
                ax4.set_ylim(yr_3d[0], yr_3d[1])

                # 2. 바닥 컨투어 생성 (로그 스케일 기반)
                if _param.shape[0] >= 3:
                    try:
                        # [핵심 수정] 정규화 시 분모를 설정한 범위 길이(x_len, y_len)로 일치시킴
                        X_train = np.column_stack(((x_raw_val - xr_3d[0])/x_len, (y_raw_val - yr_3d[0])/y_len))
                        z_log_hist = np.log10(z_raw_val + 1e-12)
                        
                        gpr_contour = GaussianProcessRegressor(kernel=C(1.0)*RBF(1.0), alpha=0.1)
                        gpr_contour.fit(X_train, z_log_hist)
                        
                        # [수정] 그리드 범위도 고정 범위와 일치시킴
                        LX_c, LY_c = np.meshgrid(np.linspace(xr_3d[0], xr_3d[1], 60), np.linspace(yr_3d[0], yr_3d[1], 60))
                        X_grid = np.column_stack(((LX_c.ravel() - xr_3d[0])/x_len, (LY_c.ravel() - yr_3d[0])/y_len))
                        Z_pred_log = gpr_contour.predict(X_grid).reshape(LX_c.shape)

                        # 3. 궤적 추적형 골짜기(Trench) 생성
                        # [수정] 거리 계산용 포인트 정규화 수치 일치
                        grid_points = np.column_stack(((LX_c.ravel()-xr_3d[0])/x_len, (LY_c.ravel()-yr_3d[0])/y_len))
                        history_points = np.column_stack(((x_raw_val-xr_3d[0])/x_len, (y_raw_val-yr_3d[0])/y_len))
                        
                        from scipy.spatial.distance import cdist
                        min_dists = np.min(cdist(grid_points, history_points), axis=1).reshape(LX_c.shape)

                        trench_depth = 2.0   
                        trench_width = 0.5  
                        
                        trench_effect = trench_depth * np.exp(-min_dists**2 / (2 * trench_width**2))
                        Z_final_contour = Z_pred_log - trench_effect

                        # 4. 컨투어 출력 (offset=0 고정)
                        cur_log_loss = np.log10(cz + 1e-12)
                        ax4.contourf(LX_c, LY_c, Z_final_contour, zdir='z', offset=0, 
                                     levels=60, cmap='rainbow', alpha=0.5,
                                     vmin=np.min(Z_final_contour), vmax=cur_log_loss + np.log10(5.0), zorder=1)
                    except Exception as e:
                        print(f"DEBUG: Canyon Error -> {e}")


                # 3. 바닥 투영 궤적 (Z=0)
                ax4.plot(x_raw_val, y_raw_val, [0]*len(_param), color='white', ls='--', lw=2.5, alpha=0.7, zorder=5)

                # 4. 실제 3D 공간 궤적 (선형 Z축 적용)
                ax4.plot(x_raw_val, y_raw_val, z_raw_val, color='black', ls='-', lw=3, alpha=0.8, zorder=10)

                # 5. 가이드라인 및 현재 최적점 (붉은 별)
                xl, yl, zl = ax4.get_xlim(), ax4.get_ylim(), ax4.get_zlim()
                ax4.plot([cx, cx], [cy, cy], [0, cz], color='red', ls=':', lw=2.5, zorder=15) 
                ax4.scatter(cx, cy, cz, color='red', marker='*', s=1500, 
                            zorder=100, edgecolors='white', linewidth=2, depthshade=False)
                
                # 6. [라벨 숫자 크기 조절] 정보 텍스트
                fixed_x, fixed_y = xl[0] + (xl[1]-xl[0])*0.05, yl[0] + (yl[1]-yl[0])*0.9
                fixed_z = zl[0] + (zl[1] - zl[0]) * 0.7
                
                # .4g를 사용하여 유효숫자 4개를 유지
                best_info = f"Stage: {s}\n\nR={cx:.4g}\nD={cy:.4g}\nL={cz:.4g}"
                ax4.text(fixed_x, fixed_y, fixed_z, best_info, 
                         color='darkred', fontweight='bold', fontsize=20, # 텍스트 크기 상향
                         zorder=110, verticalalignment='center', horizontalalignment='left')

            # 7. [라벨 숫자 크기 조절] 축 및 타이틀 설정
            ax4.set_xlabel('Learning Rate (R)', fontsize=25, fontweight='bold', labelpad=20)
            ax4.set_ylabel('Max Depth (D)', fontsize=25, fontweight='bold', labelpad=20)
            ax4.set_zlabel('MSE Loss (L)', fontsize=25, fontweight='bold', labelpad=30)
            
            # 축 눈금 숫자 크기 조절
            ax4.tick_params(axis='both', labelsize=20) 
            ax4.tick_params(axis='z', labelsize=20)
            
            ax4.set_title(f'[4] Loss Projection Landscape ({s})', fontsize=30, fontweight='bold', pad=TITLE_PAD)
            ax4.view_init(elev=25, azim=-120)
                    
            # [기본 설정] elev: 위아래 각도, azim: 좌우 회전 각도
                        
            # [핵심: 저장 및 메모리 해제]
            plt.subplots_adjust(left=0.12, right=0.88, bottom=0.1, top=0.92, wspace=0.50, hspace=0.01)
            save_path = os.path.join(specific_folder, f"{file_base_name}.png")

            # [수정 5] 저장 및 강제 종료 (메모리 부족 방지)
            plt.savefig(save_path, dpi=90) 
            plt.close(fig) # 현재 그림을 완전히 닫음
            plt.clf()      # 메모리 상의 데이터를 청소
            import gc
            gc.collect()   # 가비지 컬렉션 강제 실행 (XGB 파일 생성 보장)

    print(f" >> [Stage {s}] Files saved in XGB/GBM folders.")

print(f"\n--- 완료 ---")