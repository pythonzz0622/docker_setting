# 서버 사용 정리
https://url.kr/lnu6hj
# 서버 관리자 교육 
https://url.kr/p9j7fo
# Docker env
- Nvidia-driver : 460.91.03
- Ubuntu : 18.04
- GPU : RTX3090 
- CUDA : 11.2.1 
- CUDNN : 8
- PYTHON : 3.8
- Pytorch : 1.9.0
- Tensorflow : 2.8.0
- MMCV : 1.6.0

## Docker file을 활용한 image build
```bash
# step 1: docker build
docker build -f Dockerfile --network=host -t zz0622/{docker_image}:{tag} .
# step 2: docker run
docker run -v {folder_path}:/home/jupyter/ --network=host --gpus all -itd \
 --name {container-name} science_pack:v1
# step 3: 
docker exec -it {container-name} /bin/bash
# jupyter notebook 실행 (docker에서 실행)
jupyter notebook --allow-root --port 9000
```

## Docker hub를 활용한 image build
```
docker pull zz0622/science_pack:v1
```

## Docker hub
https://hub.docker.com/repository/docker/zz0622/science_pack

## Reference 
https://github.com/teddylee777/docker-kaggle-ko

김채민 바보
