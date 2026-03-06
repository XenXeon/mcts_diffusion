IMAGE_BASE     = cleandiffuser:base
IMAGE_DEV      = cleandiffuser:dev
WORKDIR        = /workspace
DOCKERHUB_USER = frankcholula

PIPELINE ?= veteran_d4rl_antmaze
TASK     ?= antmaze-medium-play-v2
PROJECT  ?= mcts_diffusion
NAME     ?= Default

build:
	docker build -f Dockerfile -t $(IMAGE_BASE) .
	docker build -f Dockerfile.dev -t $(IMAGE_DEV) .

run:
	docker run --gpus all -it --rm \
		-v $(PWD):$(WORKDIR) \
		-w $(WORKDIR) \
		$(IMAGE_DEV)

test:
	docker run --gpus all --rm \
		-v $(PWD):$(WORKDIR) \
		-w $(WORKDIR) \
		$(IMAGE_DEV) \
		pytest tests/test_install.py -v

train:
	docker run --gpus all -it --rm \
		-v $(PWD):$(WORKDIR) \
		-w $(WORKDIR) \
		-e WANDB_API_KEY=$(WANDB_API_KEY) \
		-e WANDB_ENTITY=$(WANDB_ENTITY) \
		$(IMAGE_DEV) \
		python pipelines/$(PIPELINE).py task=$(TASK) project=$(PROJECT) name=$(NAME)

push:
	docker tag $(IMAGE_DEV) $(DOCKERHUB_USER)/$(IMAGE_DEV)
	docker push $(DOCKERHUB_USER)/$(IMAGE_DEV)

pull:
	docker pull $(DOCKERHUB_USER)/$(IMAGE_DEV)
	docker tag $(DOCKERHUB_USER)/$(IMAGE_DEV) $(IMAGE_DEV)

.PHONY: build run test train push pull
