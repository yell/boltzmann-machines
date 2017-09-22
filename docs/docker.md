# Run in Docker container
Install [docker](https://docs.docker.com/engine/installation/linux/docker-ce) and [nvidia-docker](https://github.com/NVIDIA/nvidia-docker) if needed.

Build the project image
```bash
docker build . -t hd-models
```
Run the container (`--rm` option will automatically remove container when it exits)
```bash
nvidia-docker run -it hd-models --name=hd-models-container --rm
```
