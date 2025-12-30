import pandas as pd
from sklearn.datasets import load_wine

# 1. Scikit-learn에서 원본 와인 데이터 로드
wine = load_wine()

# 2. DataFrame 생성 시 컬럼명을 즉시 변환
# .replace('_', ' ').title()을 사용하여 'malic_acid' -> 'Malic Acid'로 변환합니다.
raw_columns = [col.replace('_', ' ').title() for col in wine.feature_names]
df = pd.DataFrame(data=wine.data, columns=raw_columns)

# 3. 타겟(레이블) 추가 (0: Wine A, 1: Wine B, 2: Wine C 등 관리에 용이)
df['Label'] = wine.target

# 4. CSV 저장 (첫 글자가 대문자인 상태로 저장됨)
df.to_csv('wine_raw_data_initial.csv', index=False)

print("--- 'wine_raw_data_initial.csv' 파일 생성이 완료되었습니다. ---")
print(f"데이터 크기: {df.shape}")
# 변환된 컬럼명 확인 (Alcohol, Malic Acid 등)
print(df[['Alcohol', 'Malic Acid', 'Label']].head())