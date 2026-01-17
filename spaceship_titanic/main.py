from data import datasetloader, data_corr_cols, one_hot, data_drop

print(f"{'='*10} Data Loading... {'='*10}")
#우주선 데이터 로드
data = datasetloader()
print(f"{'='*10} Success! {'='*10}")
print(f"{'='*10} Columns Drop... {'='*10}")
#쓰지 않을 컬럼 drop(PassengerId, Cabin,Name)
data_drop = data_drop(data)
#원 핫 인코딩으로 오브젝트형 변환
print(f"{'='*10} One Hot Encoding... {'='*10}")
data_one = one_hot(data_drop)
print(data_one.info())
print(data_one.head())
#corr로 타겟과 상관관계 확인
print(data_one.corr()['Transported'])
#절댓값 상관관계 0.2 이상인 컬럼만 추출
cols = data_corr_cols(data_one)


