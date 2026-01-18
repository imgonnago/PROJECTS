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
    df_train_drop, df_test_drop, df_test_id = drop_set_id(df_train,df_test)
    df_train_label, df_test_label = Labelencoder(df_train_drop, df_test_drop)
    cols = select_cols(df_train_label)
    df_train_label, df_test_label = scaling(df_train_label, df_test_label)
    train_x, test_x, train_y, test_y = data_split(df_train_label)
    train_x, test_x, df_test_scaled = minmax_scale(df_test_label, train_x, test_x)
    model, settings = model()
    model.fit(X_train=train_x, y_train=train_y, **settings)
    print(f"최적의 모델: {model.best_estimator}")
    check(model, test_x, test_y)
    submission(model, df_test_scaled, df_test_id)

if __name__ == "__main__":
    main()