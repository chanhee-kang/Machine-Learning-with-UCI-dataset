# Title / Machine Learning Model Test (업로드예정)
* Classification Classification Classification of Credit Approval Data Set by UCI Machine Learning Lab
* UCI data repository 중에서 credit approval data set을 선택하여 credit 유무의 이진 분류로 결정하는 문제를 기계학습모델을 이용하여 해결한다.

Credit Approval Dataset:
*  http://archive.ics.uci.edu/ml/datasets/Credit+Approval

## Installation
* Python 3.x 버젼으로 구축되어있습니다.
* Anaconda 와 Spyder 환경에서 작동하였습니다.
* 코드상 패키지를 pip명령어를 사용해 설치하여 주십시오.

## Process
* step1: Canonical Models의 Decision Tree와 Multilayer perceptron을 평가
* step2: Committee Machines의 Random Forest를 평가
* step3: Deep learning Model의 CNN모델을 평가
* step4: 학습모델의 학습 및 평가는 10-fold 평가방법을 사용한다; 즉, 전체 자료를 10등분한 후에 9 등분을 학습에 나머지 한 등분을 평가에 사용한다. 그리고 이
과정을 10번 반복한 후에 평균을 내어 각 학습모델을 평가한다
## Data Type
A1 부터 A16까지의 attribute가 존재하며 연속형 데이터와 비 연속형데이터가 존재한다. 또한 “?”의 결측 값 역시 존재한다. 또한 이진분류를 위한 target data는 +,-형태로 A16에 나타나 있다.
![uci](https://user-images.githubusercontent.com/26376653/71790545-5cd4a200-3074-11ea-8710-a2f2ffe1e130.png)
## Data Preprocessing
“Credit approval dataset”의 attribute는 총 16개로 이루어져 있으며 연속 및 불연속데이터가 혼합 되어있다. 기계학습 모델에 적용시키기 위해선 그림2 의 dataframe을 모델에 맞게 데이터 전처리 과정을 해야한다. 데이터 전처리과정은 먼저 dataframe형태로 진행되었으며, “?”로 표기된 결측치를 제거시켜 주었다. 또한 A2, A3, A8, A11, A14, A15의 연속형 데이터를 float형태로 데이터 타입을 변경하여 주었다. 속성과 클래스를 다음과 같이 분리시켜 준다. 이진 분류를 결정하는 문제이므로 A16에 존재하는 +,- 데이터를 target으로 선정한 후 0과 1의 형태로 변환 시켜 준다. 또한 학습 및 예측에 사용될 데이터를 나눠 주었다. 
![그림1](https://user-images.githubusercontent.com/26376653/71790654-e2f0e880-3074-11ea-9368-c513cbe9a53b.png)

## Step2: Cononcial Model (Decision Tree & Mulitilayer perceptron)
### Decision tree
Decision Tree (의사결정나무)는 데이터를 분석하여 데이터 사이에 존재하는 패턴을
예측 가능한 조합으로 나타낸다. 즉 tree형태로 구성된 리소스 비용 및 유틸리티를 포함한 결과를 사용하는 의사결정 지원이다.  Credit approval datasets의 이진 분류를 위해서 tree의 깊이를 3으로 설정하였고, 불순도 계산 방법으로는 entropy를 적용하여 모델을 학습시켰다.
### Multilayer perceptron
MLP (Multilayer perceptron)은 일종의 피트포워드 인공 신경망이다. MLP는 여러 층의 퍼셉트론으로 구성된 네트워크를 엄격하게 지칭한다. 먼저 MLP는 input layer, hidden layer, output layer로 총 3개의 layer로 구성된다. 이 모델에서의 parameter는 활성 함수, hidden layer의 크기 그리고 최대 학습 횟수로 구성되어 있다. 먼저 hidden layer의 크기는 총 3개의 층으로 구성 하였으며 각각 30개의 뉴런이 존재하도록 구성 하였다. 층의 뉴런의 개수를 늘리는 것보다 층의 개수를 늘리는 방법이 성능 향상에 적합하다 판단이 되어 위와 같이 구성 하였다. 또한 활성함수는 relu를 사용 하였으며 이는 일반적인 분류일 때 sigmoid나 tanh 함수에 비해 relu함수의 성능이 좋은 모습을 보였기 때문이다. 마지막으로 총 학습 횟수는 2000번으로 지정 하였다.
## Step3: Committe Machine (Random Forest)
## Step4: Deep Learning Model (CNN)
## Step5: 학습모델 평가
## Step6: 결론

## Limitations
Cononcal model과 Committee machine의 경우 Scikit-learn의 패키지 사용함. Multilayer perceptron의 경우 정확도(accuracy)가 0.6내외로 평가됨.

## Contact
작동에 문제가 생기시거나 궁금한점이 있으시면 연락주시면 감사하겠습니다 [https://ck992.github.io/](https://ck992.github.io/).
