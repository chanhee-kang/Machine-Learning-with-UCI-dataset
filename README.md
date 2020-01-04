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

## Step1: Cononcial Model (Decision Tree & Mulitilayer perceptron)
### Decision tree
 Binary classification을 하기 위해, 먼저 정규 모델의 Decision Tree 모델을 사용했다 Decision Tree의 깊이는 3으로 설정하였으며, 모델의 Overfitting을 방지하였다. 그리고 엔트로피를 불순도 계산 방법으로 적용하여 모델을 학습시켰다.
## Step2: Committe Machine (Random Forest)
## Step3: Deep Learning Model (CNN)
## Step4: 학습모델 평가
## Step5: 결론

## Limitations
Cononcal model과 Committee machine의 경우 Scikit-learn의 패키지 사용함. Multilayer perceptron의 경우 정확도(accuracy)가 0.6내외로 평가됨.

## Contact
작동에 문제가 생기시거나 궁금한점이 있으시면 연락주시면 감사하겠습니다 [https://ck992.github.io/](https://ck992.github.io/).
