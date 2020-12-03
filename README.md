# Pytorch-Yolov3-tiny-for-mobile-192x192
Pytorch dog face detection model (based on Yolov3)

모바일에서 실시간으로 구동이 가능한 Object 인식 모델을 만들고자, 최근 공부하고 구현했던 내용 정리

## 요약

Yolov3-tiny 모델에 기반했지만 Mobile을 위한 경량화를 목적으로 구조를 일부 변형,

3~5배의 속도향상(Android 구동 기준)을 시킨 object detection model 입니다. (Test & Train은 1 class로 진행했음)


## 모델구조 변형

Yolov3-tiny 모델구조를
- 전 : input image: size 416 x 416, output grid: 13x13, 26x26, Layer: 13 Conv, 6 Maxpool, 1 Upsample
- 후 : input image: size 192 x 192, output grid: 12x12, 12x12, Layer: 12 Conv, 4 Maxpool

로, 속도향상을 위해 input image size(h,w)를 1/2로 줄였고 이에 맞게 일부 레이어를 삭제 및 변형.

<br>

input image size에 맞춰 일부 레이어를 변형함에 따라, input tensor 채널 수의 forwarding에 따른 변화

3 -> 16 -> 32 -> ... -> 512 -> 1024 -> 256 ... 가

3 -> 16 -> 32 -> ... -> 512 -> 256 ... 로 변경.


## Train & Test

	main.py (Train)
	
	test.py (eval을 위한 함수)
	
	util.py (타겟 파싱, NMS 등 공용 함수)
	
	utilData.py (for dataLoader, pytorch Dataset class 이용)

- Object Class : 강아지 얼굴

- Data Sets : 총 1만장

- 강아지 얼굴 사진 4천장 RANDOM PADDING + 강아지 사진 및 얼굴 바운딩박스 라벨 4천장 + 앞의 둘을 cutMix한 4천장

- Train Data Sets : 강아지 사진 & 얼굴 바운딩박스 라벨 약 1만장

- Test(eval) Data Sets : 강아지 사진 & 얼굴 바운딩박스 라벨 400장

### Train

- pytorch adam optimizer에, step scheduler(step=10, 곱하기 0.8)를 적용하여 64 batch size로 학습.

### Test(eval)

- Output들 사이에서 iou > 0.2 일 때, Non-maximum Suprresion을 수행하고, 가장 높은 object score을 가진 bndbox만 남김.

- Target Lable bndbox와 iou > 0.5 일 때, 정답으로 처리. (한 사진에 여러 개의 bndbox가 존재할 경우, 정확도에 1/n으로 정답 및 오답 수 기록)


### 여러가지 시도들

1. Yolov3-tiny의 pre-trained weight를, 구조가 똑같은 앞부분의 레이어에 일부 적용하면 대부분 10 ~ 15 Epoch 에서 최대 정확도 달성. Scratch부터 학습 시에도 20 ~ 30 Epoch 에서 최대 정확도 달성, 하지만 pre-trained weight보단 정확도가 조금 낮음.

2. RGB와 GRAY SCALE, 두가지 방식으로 모두 학습시켜 보았고, GRAY SCALE은 RGB보다 학습속도는 빠르지만, 정확도가 약간 낮음.

3. LOSS의 크기를 결정하는 no-obj scale을 100, 50, 10, 5, 2, ... 여러가지 방식으로 학습시켜 보았지만, 기존의 100이 가장 정확도가 높음.

### 결과

- 416x416기준 최대 정확도 97%

후에 Mobile로의 적용을 위해 input image size를 줄이고 레이어를 줄인 후

- 192x192기준 최대 정확도 93%

Instagram의 #멍스타그램 사진 랜덤으로 테스트한 결과 (192x192)

<img src="https://github.com/worldbright/Pytorch-Yolov3-tiny-for-mobile-192x192-/blob/main/test_with_random_instagram_dog.png">

## Android 실시간 구동

	modelForScript.py
	
	convertScript.py

pytorch jit.trace로 모델정보 저장 후 pytorch mobile & android camerax를 이용해 android 상에서 실시간 구동

<p align="center"><img src="https://github.com/worldbright/Pytorch-Yolov3-tiny-for-mobile-192x192-/blob/main/app_%EC%A0%81%EC%9A%A9.jpg" width="25%"><img src="https://github.com/worldbright/Pytorch-Yolov3-tiny-for-mobile-192x192-/blob/main/app_%EC%A0%81%EC%9A%A92.jpg" width="25%"></p>

###### (여담 : pytorch mobile 공식 예제는 너무 구버젼의 camerax를 사용하고 있고,, 공식 android camerax 예제는 Kotlin으로 되어있어서 Java로 하려던 나한테 너무 가혹했었던..ㅠㅠ 그래도 어찌저찌 결국 최신 CameraX로, Java로 구현함. 이번 방학때 Kotlin 배워야겠다..)

CameraX로, 촬영 버튼과 자동촬영 버튼 두가지를 만들었습니다.

강아지 얼굴이 인식되면 오른쪽의 자동촬영 버튼이 빨간색으로 둘러 쌓이게 되고,

자동촬영 버튼을 누른 상태였다면 자동으로 촬영됩니다.

input image size를 192x192로 줄이기 전에는, 3 ~ 5fps였지만 현재 13 ~ 15fps로 준수한 성능을 보입니다.

## 참고자료

Yolov3-tiny 모델 레이어 구조: 
https://github.com/ultralytics/yolov3/blob/master/cfg/yolov3-tiny-1cls.cfg
  
Yolov3 Pytorch 구현:
https://github.com/eriklindernoren/PyTorch-YOLOv3
	
Android로의 model 추출을 위한 jit script의 이용은 Pytorch 공식 웹사이트를 참고했습니다.
