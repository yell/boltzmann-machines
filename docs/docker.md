# Run in a Docker container
1. Install [docker](https://docs.docker.com/engine/installation/linux/docker-ce) and [nvidia-docker](https://github.com/NVIDIA/nvidia-docker) if needed. 

2. Make sure `docker` daemon is running (`sudo service docker start`).

3. Build the project image
```bash
docker build . -t hd-models
```
4. Run the container (`--rm` option will automatically remove container when it exits)
```bash
nvidia-docker run -it hd-models --name=hd-models-container --rm
```
