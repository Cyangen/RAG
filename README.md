# LLM Stuff

Most of the updated code is in multimodal_rag

## Docker prebuilt image(outdated)

[Archive Link](https://mega.nz/file/R8Z1VaaB#j7VOCkEK-0cXwpsDkMIkNlK_rMMdFEnbZxGECJoGOBc)

## How to build and run docker
docker build -f docker/dockerfile -t test:test .

docker run -it --gpus all -p 7860:7860 -v <path_to_folder>:/LLM <image_id>