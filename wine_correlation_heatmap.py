import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# 1. 데이터 로드
df = pd.read_csv('wine_raw_data_initial.csv')

# 2. 상관관계 계산 (전체 변수 포함)
corr = df.corr()

# 3. [핵심 수정] 마스크 설정
# np.triu(..., k=1)을 사용하면 대각선(자기 자신)은 남기고 그 윗부분만 가립니다.
# 이렇게 해야 왼쪽 축 맨 위에 'Alcohol' 행이 살아납니다.
mask = np.triu(np.ones_like(corr, dtype=bool), k=1)

# 4. 시각화 설정
plt.figure(figsize=(16, 14))
sns.set_style("white")

# 5. 히트맵 그리기 (데이터 슬라이싱 없이 전체 corr 사용)
heatmap = sns.heatmap(
    corr, 
    mask=mask, 
    annot=True, 
    fmt=".2f", 
    cmap='RdBu_r', 
    center=0,
    linewidths=0.5,
    cbar_kws={"shrink": .8},
    square=True, # 모든 칸을 정사각형으로 유지
    xticklabels=corr.columns,
    yticklabels=corr.index,
    annot_kws={"size": 18, "fontweight": "bold"}
)

# 6. 라벨 및 타이틀 세부 조정
plt.title('Wine Features Correlation Matrix', fontsize=35, fontweight='bold', pad=30)
plt.xticks(rotation=45, ha='right', fontsize=12, fontweight='bold')
plt.yticks(rotation=0, fontsize=12, fontweight='bold')

# 7. 레이아웃 조정 및 저장
plt.tight_layout()
plt.savefig('wine_correlation_heatmap.png', dpi=150)
print("--- 'Alcohol'부터 'Label'까지 모두 포함된 히트맵 저장이 완료되었습니다. ---")

plt.show()