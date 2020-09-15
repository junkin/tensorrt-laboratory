BRANCH_NAME := $(shell git branch | grep \* | cut -d ' ' -f2)
BASE_IMAGE ?= nvcr.io/nvidia/tensorrt:19.05-py3
IMAGE_NAME ?= trtlab
RELEASE_IMAGE ?= ryanolson/trtlab

.PHONY: build bazel cmake tag push release clean distclean third_party vscode python

default: clean build

third_party:
	@git submodule update --init third_party/pybind11

build: third_party
	@echo FROM ${BASE_IMAGE} > .Dockerfile
	@cat Dockerfile.cmake >> .Dockerfile
	docker build -t ${IMAGE_NAME}:${BRANCH_NAME} -f .Dockerfile . 

bazel: third_party
	@echo FROM ${BASE_IMAGE} > .Dockerfile
	@cat Dockerfile.bazel >> .Dockerfile
	docker build -t ${IMAGE_NAME}-bazel:${BRANCH_NAME} -f .Dockerfile . 

python:
	@echo FROM ${BASE_IMAGE} > .Dockerfile
	@cat Dockerfile.python >> .Dockerfile
	docker build -t ${IMAGE_NAME}-python:${BRANCH_NAME} -f .Dockerfile . 

vscode: build
	docker build -t trtlab:vscode -f Dockerfile.vscode .

tag: build
	docker tag ${IMAGE_NAME} ${RELEASE_IMAGE}

push: tag
	docker push ${RELEASE_IMAGE}

release: push

clean:
	@rm -f .Dockerfile 2> /dev/null ||:
	@docker rm -v `docker ps -a -q -f "status=exited"` 2> /dev/null ||:
	@docker rmi `docker images -q -f "dangling=true"` 2> /dev/null ||:

distclean: clean
	@docker rmi ${IMAGE_NAME} 2> /dev/null ||:
	@docker rmi ${RELEASE_IMAGE} 2> /dev/null ||:
