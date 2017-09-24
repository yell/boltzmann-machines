# Run from a Docker container
## Python container
This container assumes python2 (and CUDA if needed) installed, and simply install all repo's dependencies and launch bash session.

1) Install [docker](https://docs.docker.com/engine/installation/linux/docker-ce).

2) Make sure `docker` daemon is running (`sudo service docker start`).

3) Build the project image
```bash
docker build . -t hd-models
```
4) Run the container (`--rm` option will automatically remove container when it exits)
```bash
nvidia-docker run --rm --name=hd-models-container -it hd-models bash
```

## Ubuntu-CUDA container
This container builds CUDA, cuDNN and python before installing dependencies.

1) Install additionally [nvidia-docker](https://github.com/NVIDIA/nvidia-docker).

3) Build the project image
```bash
docker build . -t hd-models-cuda -f Dockerfile.ubuntu-cuda
```
4) Run the container
```bash
nvidia-docker run --rm --name=hd-models-cuda-container -it hd-models-cuda
```
