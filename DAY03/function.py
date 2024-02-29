import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import koreanize_matplotlib
from sklearn.linear_model import LinearRegression
import numpy as np
import seaborn as sns
from sklearn.metrics import mean_squared_error,mean_absolute_error, r2_score
from sklearn.neighbors import KNeighborsRegressor
from scipy import stats
# feature : 무게, target : 길이, 데이터 비율(훈련:테스트 = 0.7:0.3)
def func(ratio=0.3, k_limit=10) :
    df=pd.read_csv("../data/fish.csv")
    test=df[["Species","Weight","Length"]]
    df=test[test["Species"].isin(["Perch"])]
    feature=df[["Weight"]]
    target=df["Length"]
    x_train, x_test, y_train, y_test = train_test_split(feature, target, test_size=ratio)
    score=dict()
    for i in range(1,3):
        for j in ['uniform','distance']:
            for s in ['auto','ball_tree','kd_tree','brute']:
                model = KNeighborsRegressor(n_neighbors=i, weights=j, algorithm=s)
                model.fit(x_train, y_train)
                y_pre=model.predict(x_test)
                score.setdefault((i,j,s),[model.score(x_train, y_train),model.score(x_test, y_test),
                                        mean_squared_error(y_test, y_pre, squared=False) , # RMSE
                                        mean_absolute_error(y_test, y_pre), # MAE
                                        mean_squared_error(y_test, y_pre),# MSE
                                        r2_score(y_test, y_pre)]) # R2
    # 성능 ,R^2, 오차제곱합, 오차절댓값합
    # 훈련점수 > 테스트 점수 : 과대적합(Overfitting)
    # 훈련점수 ≒ 테스트점수 : 최적적합
    # 훈련점수 ↓, 테스트점수 ↓ : 과소적합(Underfitting)
    max(score, key=score.get)
    res=max(score.values())
    print(f"train score : {res[0]}")
    print(f"test score : {res[1]}")
    print(f"RMSE score : {res[2]}")
    print(f"MAE score : {res[3]}")
    print(f"MSE score : {res[4]}")
    print(f"R2 score : {res[5]}")