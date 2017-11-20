# Run from a Docker container
## Python container
This container assumes python2 (and CUDA if needed) installed, and simply install all repo's dependencies and launch bash session.

1) Install [docker](https://docs.docker.com/engine/installation/linux/docker-ce).

2) Make sure `docker` daemon is running (`sudo service docker start`).

3) Build the project image
```bash
docker build . -t bm
```
4) Run the container (`--rm` option will automatically remove container when it exits)
```bash
nvidia-docker run --rm --name=bm-container -it bm bash
```

## Ubuntu-CUDA container
This container builds CUDA, cuDNN and python before installing dependencies.

1) Install additionally [nvidia-docker](https://github.com/NVIDIA/nvidia-docker).

3) Build the project image
```bash
docker build . -t bm-cuda -f Dockerfile-gpu
```
4) Run the container
```bash
nvidia-docker run --rm --name=bm-cuda-container -it bm-cuda
```
