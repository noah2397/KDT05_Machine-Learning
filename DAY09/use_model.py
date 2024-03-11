from joblib import load

# 전역 변수 
model_file="../model/iris_dt.pkl"

# 모델 로딩
model=load(model_file)

# 로딩된 모델 확인
print(model.classes_)

# 붓꽃 정보 입력 => 4개 피쳐
datas=input("붓꽃 정보 입력 (예: 꽃받침길이, 꽃받침너비, 꽃잎길이, 꽃잎너비): ")
if len(datas) :
    datas_list=list(map(float, datas.split(',')))
    print(datas_list)
    pre_iris=model.predict([datas_list])
    proba=max(model.predict_proba([datas_list])[0])
    print(f"해당 꽃은 {proba*100}% {pre_iris[0]}입니다")
else :
    print("입력된 정보가 없습니다.")