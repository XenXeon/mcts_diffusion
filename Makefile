IMAGE_BASE     = cleandiffuser:base
IMAGE_DEV      = cleandiffuser:dev
WORKDIR        = /workspace
DOCKERHUB_USER = frankcholula

PIPELINE ?= veteran_d4rl_antmaze
TASK     ?= antmaze-large-play-v2
PROJECT  ?= mcts_diffusion
GROUP    ?= antmaze
NAME     ?= Default
STEP     ?= latest
MODEL    ?= diffusion_veteran

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

eval:
	docker run --gpus all -it --rm \
		-v $(PWD):$(WORKDIR) \
		-w $(WORKDIR) \
		-e WANDB_API_KEY=$(WANDB_API_KEY) \
		-e WANDB_ENTITY=$(WANDB_ENTITY) \
		$(IMAGE_DEV) \
		python pipelines/$(PIPELINE).py \
		mode=inference task=$(TASK) project=$(PROJECT) group=$(GROUP) name=$(NAME) \
		planner_ckpt=$(STEP) critic_ckpt=$(STEP) policy_ckpt=$(STEP)

push:
	docker tag $(IMAGE_DEV) $(DOCKERHUB_USER)/$(IMAGE_DEV)
	docker push $(DOCKERHUB_USER)/$(IMAGE_DEV)

upload:
	python scripts/upload_hf.py --model $(MODEL) --task $(TASK) $(if $(STEP),--step $(STEP),)

pull:
	docker pull $(DOCKERHUB_USER)/$(IMAGE_DEV)
	docker tag $(DOCKERHUB_USER)/$(IMAGE_DEV) $(IMAGE_DEV)

.PHONY: build run test train eval push pull upload
