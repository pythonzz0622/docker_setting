# docker_setting

# Docker Env
- Ubuntu : 18.04
- GPU : RTX3090 
- CUDA : 11.2.1 
- CUDNN : 8
- PYTHON : 3.8
- PYTORCH : 1.7.1
- TENSORFLOW : 2.6.0
- MMCV : 1.6.0


```bash
# step 1: docker build
docker build -f gpu.Dockerfile --network=host -t science_pack:v3 .
# step 2: docker run
docker run -v {folder_path}:/home/jupyter/ --network=host -it --name jiwon science_pack:v4
docker --version
```

## Reference 
https://github.com/teddylee777/docker-kaggle-ko
