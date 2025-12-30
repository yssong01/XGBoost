import xgboost as xgb
try:
    # 아주 작은 데이터로 GPU 연산 테스트
    dtrain = xgb.DMatrix([[1.0]], label=[1.0])
    params = {'tree_method': 'hist', 'device': 'cuda'}
    xgb.train(params, dtrain, num_boost_round=1)
    print("✅ XGBoost GPU 가속 준비 완료!")
except Exception as e:
    print(f"!! GPU 설정 오류: {e}")