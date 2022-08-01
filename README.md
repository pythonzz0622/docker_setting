# Docker env
- Ubuntu : 18.04
- GPU : RTX3090 
- CUDA : 11.2.1 
- CUDNN : 8
- PYTHON : 3.8
- Pytorch : 1.9.0
- Tensorflow : 2.6.0
- MMCV : 1.6.0


```bash
# step 1: docker build
docker build -f Dockerfile --network=host -t {docker_image}:{tag} .
# step 2: docker run
docker run -v {folder_path}:/home/jupyter/ --network=host --gpus all -itd \
-p 8880-8889:8880-8889 --name {container-name} science_pack:v1
# step 3: 
docker exec -it science_pack:v1
# step 4:
jupyter notebook --allow-root --port 9000
```

## Docker hub
https://hub.docker.com/repository/docker/zz0622/science_pack

## Reference 
https://github.com/teddylee777/docker-kaggle-ko
