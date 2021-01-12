#!make

MAKEFILE_DIR:=$(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))
DOCKER_IMAGE=massiekm/dl-pgen:latest
CONTAINER_NAME=dl-pgen
CONTAINER_WORKSPACE=/workspace

# include .env

.DEFAULT: help

.PHONY: help build run-cpu run start stop remove attach shell

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'

build: ## Build the container for development
	docker build -f ./docker/Dockerfile -t ${DOCKER_IMAGE} ./docker/

push: ## Push the image
	docker push ${DOCKER_IMAGE}

run-cpu: ## Run development container
	docker run -d -it --name ${CONTAINER_NAME} --ipc=host -p 8888:8888 -p 6006:6006 -v ${MAKEFILE_DIR}:${CONTAINER_WORKSPACE} ${DOCKER_IMAGE}

run: ## Run the development container with gpu passthrough
	docker run --gpus all -d -it --name ${CONTAINER_NAME} --ipc=host -p 8888:8888 -p 6006:6006 -v ${MAKEFILE_DIR}:${CONTAINER_WORKSPACE} ${DOCKER_IMAGE}

start: ## Start the development container
	docker start ${CONTAINER_NAME}

stop: ## Stop the development container
	docker stop ${CONTAINER_NAME}

remove: ## Remove the development container
	docker rm ${CONTAINER_NAME}

attach: ## Attach to running container
	docker attach ${CONTAINER_NAME}

shell: ## Attach to container with bash
	docker exec -it ${CONTAINER_NAME} /bin/bash