from data import datasetloader, drop_set_id, Labelencoder,select_cols, scaling,data_split,minmax_scale
from model import  model
from Util import EDA, check, submission

class main():
    print(f"{'='*10} Data Loading... {'='*10}")
    #우주선 데이터 로드
    df_train, df_test = datasetloader()
    print(f"{'='*10} Success! {'='*10}")
    print(f"{'='*10} Columns Drop... {'='*10}")
    #쓰지 않을 컬럼 drop(PassengerId, Cabin,Name)
    #submission에서 사용할 id는 따로 남겨두기
    df_train_drop, df_test_drop, df_test_id = drop_set_id(df_train,df_test)
    #labelencoder로 object형 숫자형으로 변환
    print(f"{'=' * 10} Label Encoding... {'=' * 10}")
    df_train_label, df_test_label = Labelencoder(df_train_drop, df_test_drop)
    #corr 0.2이상인 컬럼 선택
    print(f"{'=' * 10} Select Columns (abs(corr) > 0.2) ... {'=' * 10}")
    cols = select_cols(df_train_label)
    #결측치 처리-> 최빈값으로 대체
    print(f"{'=' * 10} fillna (mode) ... {'=' * 10}")
    df_train_label, df_test_label = scaling(df_train_label, df_test_label)
    #trian, test셋 나누기
    print(f"{'=' * 10} train, test split ... {'=' * 10}")
    train_x, test_x, train_y, test_y = data_split(df_train_label)
    #마지막으로 minmax scale 해주기
    print(f"{'=' * 10} Min Max scale... {'=' * 10}")
    train_x, test_x, df_test_scaled = minmax_scale(df_test_label, train_x, test_x)
    #모델 생성 및 학습
    print(f"{'=' * 10} AutoML ... {'=' * 10}")
    model, settings = model()
    print(f"{'=' * 10} Model fitting... {'=' * 10}")
    model.fit(X_train=train_x, y_train=train_y, **settings)
    print(f"최적의 모델: {model.best_estimator}")
    check(model, test_x, test_y)
    #submission 생성
    print(f"{'=' * 10} Submission creat ... {'=' * 10}")
    submission(model, df_test_scaled, df_test_id)

if __name__ == "__main__":
    main()