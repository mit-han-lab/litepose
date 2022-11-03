LitePose for internal use
=========================

## 1. 지원 범위

- Litepose XS
- Litepose S
- Litepose M
- Litepose L
- HigherHRNet (w32)

<br/>

## 2. 성능

|model|FLOPs(GMac)|# of parameters|mAP(0.5)|mAP(0.5:0.9)|
|---|---|---|---|---|
|LitePose-XS|
|LitePose-S|4.98|2.73|0.744|0.514|
|LitePose-M|7.84|3.53|0.759|0.517|
|LitePose-L|
|HigherHRNet (w32)|48.03|28.64|0.822|0.601|

* coco pretrained model 에 crwod pose 데이터셋 학습 결과

<br/>



## 3. 사용 방법
1. 데이터셋 준비
- COCO 형식의 데이터셋과 CrowdPose 형식의 데이터셋 사용 가능
- COCO 의 경우 annotation 파일은 annotation 폴더 아래, image 파일은 images 폴더 아래에 분류 별로 저장한다.
> dataset/  
> dataset/annotation/  
> dataset/images/  
> dataset/images/test2017/  
> dataset/images/train2017/  
> dataset/images/val2017/
- CrowdPose 의 경우 annotation 파일은 json 폴더 아래, images 폴더 아래 분류 상관 없이 저장한다.
> dataset/  
> dataset/json/  
> dataset/images/  

<br/>

2. git clone
```
git clone https://github.com/jwbaek-nota/litepose.git
```
<br/>

3. docker build

```
docker build -f Dockerfile -t notadockerhub/litepose:latest .
```

<br/>

4. docker run  
host 의 데이터셋이 있는 위치와 결과 파일들을 저장할 위치를 mount  시켜준다. 아래 예시에서는 host 의 ~/dataset 폴더에 데이터셋이 저장되어 있고 ~/output 폴더에 결과 파일들이 저장될 것이다.

```
docker run --gpus '"device=0,1,2,3"' --shm-size=8G -it -v ~/dataset:/root/dataset -v ~/output:/root/output notadockerhub/litepose:latest bash
```

<br/>

5. 기본 세팅으로 학습  
```
python main.py --json json_examples/litepose_l_crowdpose.json
```
* LitePose XS : json_examples/litespose_xs_crowdpose.json
* LitePose S : json_examples/litespose_s_crowdpose.json
* LitePose M : json_examples/litespose_m_crowdpose.json
* LitePose L : json_examples/litespose_l_crowdpose.json
* HigherHRNet (w32) : json_examples/higherhrnet_crowdpose.json

<br/>

6. 커스텀 세팅으로 학습 시 유의사항
* json 파일을 수정하여 커스텀 세팅으로 학습이 가능함
* 필수
```
model_type : litepose_xs, litepose_s, litepose_m, litepose_l, higherhrnet_w32 중 1  

dataset_path : container 내의 dataset 의 위치

num_joint : keypoint (joint) 의 갯수

dataset_format : crowd_pose, coco 중 1

output_path : container 내에서 결과 파일들이 저장될 위치 - host 에 mounting 하는 것을 권장
```
* 선택
```
cfg - 기타 세팅
```
cfg 하위에 정의된 값들은 experiments/crowd_pose/mobilenet/mobile.yaml 파일에 덮어씌워진다. (higherhrnet 의 경우에는 experiments/crowd_pose/higher_hrnet/w32_512_adam_lr1e-3.yaml 파일)  

예시 : Training epoch 을 커스텀 세팅하고 싶은 경우, 이 값은 mobile.yaml 의 TRAIN / END_EPOCH 에 정의되어 있으므로 아래와 같이 동일한 레벨에 정의를 해준다.
```
{
  "model_type" : "litepose_xs",
  "dataset_path" : "/root/dataset/crowdpose",
  "num_joint" : 14,
  "dataset_format" : "crowd_pose",
  "cfg": {
      "TRAIN": {
          "END_EPOCH": 500
      }
  },
  "output_path" : "/root/output"
}
```

7. 기타

* log (tensorboard) 는 {output_path}/{task_id}/log 에 저장되며, host 에 바로 저장되므로 host 에서 tensorboard 서버를 실행하여 진행 상황을 확인하는 것이 가능하다.
* output (pth 파일, onnx 파일, 실행 결과 데이터 등등) 은 {output_path}/{task_id}/output 에 저장되며, 이 결과물은 실행이 완료 된 후에 host 에서 확인이 가능하다.