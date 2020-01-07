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
* step1: 데이터 preprocessing
* step2: Canonical Models: Decision Tree와 Multilayer perceptron을 평가
* step3: Committee Machines: Random Forest를 평가
* step4: Deep learning Model: CNN모델을 평가
* step5: 학습모델의 학습 및 평가: 10-fold 평가방법을 사용
* step6: 기계학습모델 평가 비교분석
* step7: 결론

## Data Type
A1 부터 A16까지의 attribute가 존재하며 연속형 데이터와 비 연속형데이터가 존재한다. 또한 “?”의 결측 값 역시 존재한다. 또한 이진분류를 위한 target data는 +,-형태로 A16에 나타나 있다.
![uci](https://user-images.githubusercontent.com/26376653/71790545-5cd4a200-3074-11ea-8710-a2f2ffe1e130.png)
## Data Preprocessing
“Credit approval dataset”의 attribute는 총 16개로 이루어져 있으며 연속 및 불연속데이터가 혼합 되어있다. 기계학습 모델에 적용시키기 위해선 그림2 의 dataframe을 모델에 맞게 데이터 전처리 과정을 해야한다. 데이터 전처리과정은 먼저 dataframe형태로 진행되었으며, “?”로 표기된 결측치를 제거시켜 주었다. 

또한 A2, A3, A8, A11, A14, A15의 연속형 데이터를 float형태로 데이터 타입을 변경하여 주었다. 속성과 클래스를 다음과 같이 분리시켜 준다. 이진 분류를 결정하는 문제이므로 A16에 존재하는 +,- 데이터를 target으로 선정한 후 0과 1의 형태로 변환 시켜 준다. 또한 학습 및 예측에 사용될 데이터를 나눠 주었다.

![그림1](https://user-images.githubusercontent.com/26376653/71790654-e2f0e880-3074-11ea-9368-c513cbe9a53b.png)

## Step2: Cononcial Model
### 2.1 Decision tree
Decision Tree (의사결정나무)는 데이터를 분석하여 데이터 사이에 존재하는 패턴을
예측 가능한 조합으로 나타낸다. 즉 tree형태로 구성된 리소스 비용 및 유틸리티를 포함한 결과를 사용하는 의사결정 지원이다.  Credit approval datasets의 이진 분류를 위해서 tree의 깊이를 3으로 설정하였고, 불순도 계산 방법으로는 entropy를 적용하여 모델을 학습시켰다.
### 2.2 Multilayer perceptron
MLP (Multilayer perceptron)은 일종의 피트포워드 인공 신경망이다. MLP는 여러 층의 퍼셉트론으로 구성된 네트워크를 엄격하게 지칭한다. 먼저 MLP는 input layer, hidden layer, output layer로 총 3개의 layer로 구성된다. 이 모델에서의 parameter는 활성 함수, hidden layer의 크기 그리고 최대 학습 횟수로 구성되어 있다. 먼저 hidden layer의 크기는 총 3개의 층으로 구성 하였으며 각각 30개의 뉴런이 존재하도록 구성 하였다. 

층의 뉴런의 개수를 늘리는 것보다 층의 개수를 늘리는 방법이 성능 향상에 적합하다 판단이 되어 위와 같이 구성 하였다. 또한 활성함수는 relu를 사용 하였으며 이는 일반적인 분류일 때 sigmoid나 tanh 함수에 비해 relu함수의 성능이 좋은 모습을 보였기 때문이다. 마지막으로 총 학습 횟수는 2000번으로 지정 하였다.
## Step3: Committe Machine
### 3.1 Random Forest
Random Forest는 앙상블 방법으로 여러 개의 decision tree로 이루어져 있다. 먼저 random forest를 구현하기 위해선 많은 양의 decision tree가 필요하며 각각의 tree는 target에 대한 예측을 잘 수행되어야 하며 다른 tree와는 구별되어야 한다. 또한 각각의 tree는 독립적으로 만들어져야 하며 알고리즘은 각 tree가 고유하게 만들어지기 위해 무작위 선택을 한다. 따라서 tree를 구축하기 전에 dataset의 bootstrap sample 및 전체 dataset중에서 무작위로 데이터를 전체 데이터의 개수만큼 중복 및 반복을 하여 샘플을 만든다. 

그리고 이 샘플 데이터에 기초하여 decision tree를 생성한다. Scikit-learn이 제공하는 random forest API는 bootstrap 샘플의 크기 n의 값으로 기존 training dataset의 전체 개수와 같은 수를 할당 하여주며, 데이터 특성 값을 중복 허용 없이 위해 필요한 값을 데이터 전체 특성의 개수의 제곱근으로 할당하여 준다. 이 모델에서 선정한 parameter 는 생성할 decision tree, tree의 깊이, 불순도 계산 방법이다. 해당 모델에서는 총 100개의 decision tree를 생성하였으며 over fitting 방지를 위해 tree의 최대 깊이를 3으로 선정하고 entropy를 불순도 계산방법으로 선정 하였다.
## Step4: Deep Learning Model
### 4.1 CNN
CNN(Convolutional Neural Network)은 MLP에 합성곱 계층과 풀링 계층이라는 고유의 구조를 더한 Neural Network이라고 할 수 있다. 여기서 합성곱 계층은 필터, 혹은 커널이라고 하는 작은 수용 영역을 통해 데이터를 인식하는 계층이고, 풀링 레이어는 특정 영역에서 최대값만 추출하거나, 평균값을 추출하여 차원을 축소하는 역할을 한다. CNN은 MLP에 비해 학습해야 할 파라 미터의 개수가 상대적으로 적어 학습이 빠르다는 장점이 있다. 

위의 데이터 셋의 이진 분류를 위한 CNN 모델을 3개의 convolution 계층과 3개의 pooling 계층을 가진 구조를 생성했다. 그리고 이진 분류를 위해 활성화 함수로 LeakyReLU로 설정하였고, 각 convolution 계층 마다 필터의 개수는 32, 64, 128개, 그리고 필터의 사이즈는 모두 (3,3)으로 설정하였다. 그리고 pooling 계층의 pool 사이즈는 모두 (2,2)로 설정하였다. 모델의 요약 구조는 아래와 같다.
![image](https://user-images.githubusercontent.com/26376653/71864246-a7245480-3142-11ea-92d9-f2920848fc07.png)
## Step5: 학습모델 평가
### 5.1 K-fold 평가기법(k=10)
각 학습모델의 학습 및 평가는 10-fold 평가 방법을 사용 하였다. 전체 자료를 10등분한 후에 9 등분을 학습에 나머지 한 등분을 평가에 사용하며 이 과정을 10번 반복한 후에 평균을 내어 각 모델을 평가 하는 기법이다.  또한 각 모델의 평가의 측도로 Accuracy, Precision, Recall, 그리고 F-1 score를 사용하였다.

![image](https://user-images.githubusercontent.com/26376653/71865253-890c2380-3145-11ea-917d-42fb017afb4d.png)

## Step6: 평가 결과 비교 분석
Accuracy와 Precision이 제일 높게 나온 모델은 각각 0.902, 0.855로 Random Forest 모델이며 Recall 과 F-1 Score의 경우 CNN 모델이 각각 0.870, 0.83으로 제일 높은 수치를 보여주었다. 총 4가지 모델 중 Multilayer perceptron 모델의 평가가 다른 모델에 비해 낮은 수치를 보여 주었으며 Multilayer perceptron 과 CNN 모델이 타 모델에 비해 높은 수치를 보여 주었다. 10-fold 기법을 사용하여 평가를 내보았을 때, Random Forest, CNN, Decision Tree 모델을 사용하는 것이 UCI dataset의 일부인 credit approval dataset을 이진분류하기에 적합한 모델이라 판단된다. 

|  | 1 | 2 | 3 | 4 |
|:--------:|:--------:|:--------:|:--------:|:--------:|
| Accuracy | Random Forest | CNN | Decision Tree | MLP |
| Precision | Random Forest | CNN | Decision Tree | MLP |
| Recall | CNN | Random Forest | Decision Tree | MLP |
| F1 Score | CNN | Random Forest | Decision Tree | MLP |

먼저 decision tree는 흔히 의사 결정 트리 라 고 불리는데 이름에서 알 수 있듯이 주로 데이터를 분류 하는데 용이하게 사용된다. 즉, Credit approval dataset 에서 +와 –로 나누어 지는 credit이진 분류를 결정하기 적합하다. 또한 decision tree는 데이터가 특정 범위 안에 들어오도록 하는 정규화 혹은 표준화 같은 데이터 전처리 과정이 크게 필요 없으며 이진 특성과 연속적인 특성이 혼합되어 있을 때도 좋은 성능을 보인다. 본 리포트에선 정확도 0.782으로 높은 수치를 보여주었다. 하지만 decision tree의 단점 역시 존재한다. 단점으로는 사전 가지치기를 사용함에도 over fitting이 되는 경향이 있으며 모델의 일반화 성능이 좋지만은 않다. 따라서 이러한 decision tree의 단점을 보안하기 위하여 decision tree의 앙상블 방법을 사용 할 수 있다.

앙상블은 여러 기계학습 모델을 합치어 성능 높은 모델을 만드는 것이다. 어떠한 dataset이 주어졌을 때, dataset을 나누어 각각 학습시킨 뒤 모델을 합쳐 전체 데이터의 결과를 산출한다. Random forest모델은 학습에 있어서 다수의 decision tree를 구성하고 다수의 decision tree로부터 분류 혹은 회귀분석 결과를 출력함으로써 동작된다. 따라서 decision tree의 문제인 training data에 over fitting 문제를 해결할 수 있다. Random frost의 경우 decision tree에 비해 accuracy, precision, recall, F-1 score방면에서 약 10% 성능이 향상 된 것을 확인 할 수 있다. 

MLP 모델은 입력 층에서 전달되는 값이 은닉 층의 모든 노드로 전달되고 은닉 층 노드의 값은 다시 출력 층 노드로 전달되는 구조인 순전 파 형식을 가진다. 결과에서는 먼저 트레이닝 셋이 과 최적화로 인해 실제 credit approval datasets을 넣어 분류를 하면 정확도가 떨어지는 over fitting 문제와 layer가 깊어지면서 역 전파로 인해 에러를 뒤로 전파하게 되는 데에 문제가 생긴 것으로 보인다.

CNN모델은 위의 ML의 문제를 해결할 수 있다. 먼저 overfitting문제의 경우 정규화를 통해 weight이 너무 커버리지 않게 조절을 할 수 있으며 활성화 함수를 변경함에 있어서 평가의 결과가 더 좋게 나온 것으로 판단된다.

## Step7: 결론
기계학습의 3가지 모델인 canonical model, committee machine, deep learning모델을 사용하여 UCI dataset의 credit approval datasets을 이진 분류로 평가하였다. 해당 dataset은 신용카드 어플리케이션과 관련이 있어 개인정보 보호를 위하여 모든 속성값이 의미 없는 기호로 변경이 되어있었다. 또한 결측 값 및 모델에 맞춰 전처리 과정을 거쳐야 하였다. 

기계학습 모델에 적용을 위해 연속형 데이터 전처리, target data 전처리등 을 일괄적으로 거쳤으며 각 모델별로 필요한 전처리과정을 다시 거쳤다. Canonical model에선 decision tree 와 multilayer perceptron을 평가하였고 committee machine에선 random forest를 평가하였으며 deep learning 모델에서는 CNN을 평가하였으며 10-fold방식으로 각 4가지 모델을 최종 평가 하였다. 최종 10-fold 결과를 분석하면 이진분류에서는  random forest, cnn, 그리고 decision tree의 성능이 높게 나왔다. 

정확히 비교하자면 random forest – cnn – decision tree – multilayer perceptron의 순으로 나왔지만 앞 3가지 모델은 정확도 0.7이상을 보여주어 평가의 정확도가 높게 나왔다. Multilayer perceptron의 경우 over fitting 문제와 역전파 현상으로 정확도가 0.6의 수치를 보였지만 추가 모델 개발을 통해 성능을 향상시킬 여지가 있어 보인다.
## 참고문헌
Bhukya, D. and Ramachandram, S. (2010). Decision Tree Induction: An Approach for Data Classification Using AVL-Tree. International Journal of Computer and Electrical Engineering, pp.660-665.

Chen, L. and Tang, H. (2004). Improved computation of beliefs based on confusion matrix for combining multiple classifiers. Electronics Letters, 40(4), p.238.

Fourie, C. (2003). Deep learning? What deep learning?. South African Journal of Higher Education, 17(1).
Koo, I., Lee, N. and Kil, R. (2008). Parameterized cross-validation for nonlinear regression models. Neurocomputing, 71(16-18), pp.3089-3095.

Mantas, C., Castellano, J., Moral-García, S. and Abellán, J. (2018). A comparison of random forest based algorithms: random credal random forest versus oblique random forest. Soft Computing, 23(21), pp.10739-10754.
 Mühlenbein, H. (1990). Limitations of multi-layer perceptron networks - steps towards genetic neural networks. Parallel Computing, 14(3), pp.249-260.
 
Uchida, K., Tanaka, M. and Okutomi, M. (2018). Coupled convolution layer for convolutional neural network. Neural Networks, 105, pp.197-205.
## Limitations

## Contact
작동에 문제가 생기시거나 궁금한점이 있으시면 연락주시면 감사하겠습니다 [https://ck992.github.io/](https://ck992.github.io/).
