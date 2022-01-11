# Image_Classifier
이미지 인식 모델 학습시키기
Pretrained Model 획득을 목적으로, ImageNet 데이터셋으 학습시켜 결과를 확인해본다.

# Requirements
torch
torchinfo
matplotlib
numpy

# How to Use

CMD창에서 모든 작업이 이루어지며 입력값을 받아서 학습을 진행한다.

1. ImageNet 데이터셋 폴더 경로 불러오기
  -> Training, Validation 폴더 경로를 불러온다.

2. 데이터셋 관련 세부설정하기
  -> Batch Size 입력
  -> DataLoader 내 Num_worker 수 입력

3. 모델 불러오기
  -> 숫자를 입력하여 모델을 불러온다.

4. Loss Function 설정하기
  -> Default : CrossEntropyLoss
  
5. Optimizer 설정하기
  -> Learning Rate 입력
  -> Momentum 입력
  -> Weight Decay 입력
  -> 숫자를 입력하여 Optimizer를 불러온다.
  
6. 학습하기

7. 결과확인하기


# Training Result

## 1. ResNet-18

Training  
  Top-1 Accuracy : 70.934%  
  Top-5 Accuracy : 88.390%  

Validation  
  Top-1 Accuracy : 70.436%  
  Top-5 Accuracy : 89.678%  