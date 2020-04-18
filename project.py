import numpy as np
import keras

from pandas import DataFrame as df
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_validate
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
from keras.utils import to_categorical
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dense, Flatten
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Sequential

# 데이터 로드
with open('crx.data') as file:
    data=[]
    for line in file.readlines():
        data.append(line.split(','))
    for i in range(len(data)):
        data[i][0]=data[i][0][len(data[i][0])-1]
        data[i][len(data[i])-1]=data[i][len(data[i])-1][0]
df1=df(data,columns=['A1','A2','A3','A4','A5','A6','A7','A8','A9','A10','A11','A12','A13','A14','A15','A16'])

#데이터 전처리 (for DLT) 
#target-> A16 ('-': negative -> 0, '+': positive -> 1)
df1.A16=df1.A16.replace('-',0)
df1.A16=df1.A16.replace('+',1)

df1.A1=df1.A1.replace('a',2)
df1.A1=df1.A1.replace('b',3)

df1.A4=df1.A4.replace('u',4)
df1.A4=df1.A4.replace('y',5)
df1.A4=df1.A4.replace('l',6)
df1.A4=df1.A4.replace('t',7)

df1.A5=df1.A5.replace('g',8)
df1.A5=df1.A5.replace('p',9)
df1.A5=df1.A5.replace('gg',10)

df1.A6=df1.A6.replace('c',11)
df1.A6=df1.A6.replace('d',12)
df1.A6=df1.A6.replace('cc',13)
df1.A6=df1.A6.replace('i',14)
df1.A6=df1.A6.replace('j',15)
df1.A6=df1.A6.replace('k',16)
df1.A6=df1.A6.replace('m',17)
df1.A6=df1.A6.replace('r',18)
df1.A6=df1.A6.replace('q',19)
df1.A6=df1.A6.replace('w',20)
df1.A6=df1.A6.replace('x',21)
df1.A6=df1.A6.replace('e',22)
df1.A6=df1.A6.replace('aa',23)
df1.A6=df1.A6.replace('ff',24)

df1.A7=df1.A7.replace('v',25)
df1.A7=df1.A7.replace('h',26)
df1.A7=df1.A7.replace('bb',27)
df1.A7=df1.A7.replace('j',28)
df1.A7=df1.A7.replace('n',29)
df1.A7=df1.A7.replace('z',30)
df1.A7=df1.A7.replace('dd',31)
df1.A7=df1.A7.replace('ff',32)
df1.A7=df1.A7.replace('o',33)

df1.A9=df1.A9.replace('t',34)
df1.A9=df1.A9.replace('f',35)

df1.A10=df1.A10.replace('t',36)
df1.A10=df1.A10.replace('f',37)

df1.A12=df1.A12.replace('t',38)
df1.A12=df1.A12.replace('f',39)

df1.A13=df1.A13.replace('g',40)
df1.A13=df1.A13.replace('p',41)
df1.A13=df1.A13.replace('s',42)

#결측치 처리
df1=df1.replace('?',np.nan)
df1=df1.dropna(how='any')

#데이터 타입 변경
df1=df1.astype({'A2':'float','A3':'float','A8':'float','A11':'float','A14':'float','A15':'float'})

#속성과 클래스 분리
X=np.array(df1.loc[:,'A1':'A15'])
y=np.array(df1.loc[:,'A16'])

#학습&예측 데이터 나누기
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.1)

#데이터 학습(DT)
dt=DecisionTreeClassifier(random_state=42)
dt=dt.fit(X_train, y_train)
print("DT 훈련 세트 정확도: {:.3f}".format(dt.score(X_train, y_train)))
print("DT 테스트 세트 정확도: {:.3f}".format(dt.score(X_test, y_test)))

#데이터 전처리 (for MLP)
mean_on_train=X_train.mean(axis=0)
std_on_train=X_train.std(axis=0)
X_train_scaled = (X_train - mean_on_train) / std_on_train
X_test_scaled = (X_test - mean_on_train) / std_on_train

#데이터 학습(MLP)
mlp=MLPClassifier(solver='adam',max_iter=2000,random_state=42,hidden_layer_sizes=(200,100))
mlp=mlp.fit(X_train,y_train)
print("MLP 훈련 세트 정확도: {:.3f}".format(mlp.score(X_train_scaled, y_train)))
print("MLP 테스트 세트 정확도: {:.3f}".format(mlp.score(X_test_scaled, y_test)))

#데이터 학습(RandomForest)
forest=RandomForestClassifier(n_estimators=100,random_state=42)
forest=forest.fit(X_train,y_train)
print("Random Forest 훈련 세트 정확도: {:.3f}".format(forest.score(X_train, y_train)))
print("Random Forest 테스트 세트 정확도: {:.3f}".format(forest.score(X_test, y_test)))

#데이터 전처리 (for CNN)
y_cnn=to_categorical(y_train)
scaler=MinMaxScaler(feature_range=(0,1))
scaler.fit(X_train)
X_train_cnn=scaler.transform(X_train)
X_train_cnn=X_train_cnn.reshape(-1,15,1,1)

#데이터 학습(CNN)
batch_size=64
epochs=50
num_classes=2
cv = KFold(10, shuffle=True, random_state=0)
accs=[]
precs=[]
recalls=[]
f1s=[]
for i, (idx_train, idx_test) in enumerate(cv.split(X_train_cnn)):
    print(i)
    dataset_x_train=X_train_cnn[idx_train]
    dataset_x_test=X_train_cnn[idx_test]
    dataset_y_train=y_cnn[idx_train]
    dataset_y_test=y_cnn[idx_test]
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),activation='linear',input_shape=(15,1,1),padding='same'))
    model.add(LeakyReLU(alpha=0.1))
    model.add(MaxPooling2D((2, 2),padding='same'))
    model.add(Conv2D(64, (3, 3), activation='linear',padding='same'))
    model.add(LeakyReLU(alpha=0.1))
    model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
    model.add(Conv2D(128, (3, 3), activation='linear',padding='same'))
    model.add(LeakyReLU(alpha=0.1))                  
    model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
    model.add(Flatten())
    model.add(Dense(10, activation='linear'))
    model.add(LeakyReLU(alpha=0.1))                  
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(),metrics=['accuracy'])
    model.fit(dataset_x_train, dataset_y_train, batch_size=batch_size,epochs=epochs,verbose=1)
    real=dataset_y_test
    pred=np.argmax(model.predict(dataset_x_test),1)
    accuracy = np.mean(np.equal(np.argmax(real,1),pred))
    accs.append(accuracy)
    right = np.sum(np.argmax(real,1) * pred == 1)
    precision = right / np.sum(pred)
    precs.append(precision)
    recall = right / np.sum(np.argmax(real,1))
    recalls.append(recall)
    f1 = 2 * precision*recall/(precision+recall)
    f1s.append(f1)

#K-Fold 평가
scoring = {'accuracy' : make_scorer(accuracy_score), 
           'precision' : make_scorer(precision_score),
           'recall' : make_scorer(recall_score), 
           'f1_score' : make_scorer(f1_score)}

kfold=KFold(n_splits=10,shuffle=True,random_state=42)
result_DT=cross_validate(estimator=dt,X=X_test,y=y_test,cv=kfold,scoring=scoring)
result_mlp=cross_validate(estimator=mlp,X=X_test,y=y_test,cv=kfold,scoring=scoring)
result_forest=cross_validate(estimator=forest,X=X_test,y=y_test,cv=kfold,scoring=scoring)

print("\nDecision Tree")
print("DT model Accuracy: {:.3f}".format(result_DT['test_accuracy'].mean()))
print("DT model Precision: {:.3f}".format(result_DT['test_precision'].mean()))
print("DT model Recall: {:.3f}".format(result_DT['test_recall'].mean()))
print("DT model F1: {:.3f}".format(result_DT['test_f1_score'].mean()))
print("\nMulti layer Perceptron")
print("MLP model Accuracy: {:.3f}".format(result_mlp['test_accuracy'].mean()))
print("MLP model Precision: {:.3f}".format(result_mlp['test_precision'].mean()))
print("MLP model Recall: {:.3f}".format(result_mlp['test_recall'].mean()))
print("MLP model F1: {:.3f}".format(result_mlp['test_f1_score'].mean()))
print("\nRandom Froest")
print("Random forest model Accuracy: {:.3f}".format(result_forest['test_accuracy'].mean()))
print("Random forest model Precision: {:.3f}".format(result_forest['test_precision'].mean()))
print("Random forest model Recall: {:.3f}".format(result_forest['test_recall'].mean()))
print("Random forest model F1: {:.3f}".format(result_forest['test_f1_score'].mean()))
print("\nCNN")
print("CNN model Accuracy: {:.3f}".format(np.mean(accs)))
print("CNN model Precision: {:.3f}".format(np.mean(precs)))
print("CNN model Recall: {:.3f}".format(np.mean(recalls)))
print("CNN model F1: {:.3f}".format(np.mean(f1s)))