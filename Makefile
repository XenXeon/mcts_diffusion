IMAGE_BASE     = cleandiffuser:base
IMAGE_DEV      = cleandiffuser:dev
WORKDIR        = /workspace
DOCKERHUB_USER = frankcholula
D4RL_CACHE     ?= $(HOME)/.d4rl

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

# Unit tests for the MCTS expansion primitive (CPU only, no checkpoint needed)
test-mcts-unit:
	docker run --rm \
		-v $(PWD):$(WORKDIR) \
		-w $(WORKDIR) \
		$(IMAGE_DEV) \
		pytest tests/test_mcts_expansion.py -m "not integration" -v

# Full MCTS tests including integration tests (requires checkpoint + GPU)
test-mcts:
	docker run --gpus all --rm \
		-v $(PWD):$(WORKDIR) \
		-v $(D4RL_CACHE):/root/.d4rl \
		-w $(WORKDIR) \
		$(IMAGE_DEV) \
		pytest tests/test_mcts_expansion.py -v

# Smoke test: one real expansion end-to-end (requires checkpoint + GPU)
smoke-phase2:
	docker run --gpus all --rm \
		-v $(PWD):$(WORKDIR) \
		-v $(D4RL_CACHE):/root/.d4rl \
		-w $(WORKDIR) \
		$(IMAGE_DEV) \
		python scripts/phase2_smoke_test.py

train:
	docker run --gpus all -it --rm \
		-v $(PWD):$(WORKDIR) \
		-v $(D4RL_CACHE):/root/.d4rl \
		-w $(WORKDIR) \
		-e WANDB_API_KEY=$(WANDB_API_KEY) \
		-e WANDB_ENTITY=$(WANDB_ENTITY) \
		$(IMAGE_DEV) \
		python pipelines/$(PIPELINE).py task=$(TASK) project=$(PROJECT) name=$(NAME)

eval:
	docker run --gpus all -it --rm \
		-v $(PWD):$(WORKDIR) \
		-v $(D4RL_CACHE):/root/.d4rl \
		-w $(WORKDIR) \
		-e WANDB_API_KEY=$(WANDB_API_KEY) \
		-e WANDB_ENTITY=$(WANDB_ENTITY) \
		$(IMAGE_DEV) \
		python pipelines/$(PIPELINE).py \
		mode=inference task=$(TASK) project=$(PROJECT) group=$(GROUP) name=$(NAME) \
		planner_ckpt=$(STEP) critic_ckpt=$(STEP) policy_ckpt=$(STEP)

render:
	docker run --gpus all -it --rm \
		-v $(PWD):$(WORKDIR) \
		-v $(D4RL_CACHE):/root/.d4rl \
		-w $(WORKDIR) \
		-e WANDB_API_KEY=$(WANDB_API_KEY) \
		-e WANDB_ENTITY=$(WANDB_ENTITY) \
		$(IMAGE_DEV) \
		python pipelines/$(PIPELINE).py \
		mode=render task=$(TASK) project=$(PROJECT) name=$(NAME) \
		planner_ckpt=$(STEP) critic_ckpt=$(STEP) policy_ckpt=$(STEP)

# Unit tests for the MCTS tree (CPU only, no checkpoint needed)
test-mcts-tree-unit:
	docker run --rm \
		-v $(PWD):$(WORKDIR) \
		-w $(WORKDIR) \
		$(IMAGE_DEV) \
		pytest tests/test_mcts_tree.py -m "not integration" -v

# Full MCTS tree tests including integration tests (requires checkpoint + GPU)
test-mcts-tree:
	docker run --gpus all --rm \
		-v $(PWD):$(WORKDIR) \
		-v $(D4RL_CACHE):/root/.d4rl \
		-w $(WORKDIR) \
		$(IMAGE_DEV) \
		pytest tests/test_mcts_tree.py -v

# Phase 3 ablation: compare all three storage modes (requires checkpoint + GPU)
ablation-phase3:
	docker run --gpus all --rm \
		-v $(PWD):$(WORKDIR) \
		-v $(D4RL_CACHE):/root/.d4rl \
		-w $(WORKDIR) \
		$(IMAGE_DEV) \
		python scripts/phase3_ablation.py

# Phase 3 K ablation: vary K with fixed total evaluation budget (requires checkpoint + GPU)
k-ablation-phase3:
	docker run --gpus all --rm \
		-v $(PWD):$(WORKDIR) \
		-v $(D4RL_CACHE):/root/.d4rl \
		-w $(WORKDIR) \
		$(IMAGE_DEV) \
		python scripts/phase3_k_ablation.py

push:
	docker tag $(IMAGE_DEV) $(DOCKERHUB_USER)/$(IMAGE_DEV)
	docker push $(DOCKERHUB_USER)/$(IMAGE_DEV)

upload:
	python scripts/upload_hf.py --model $(MODEL) --task $(TASK) $(if $(STEP),--step $(STEP),)

pull:
	docker pull $(DOCKERHUB_USER)/$(IMAGE_DEV)
	docker tag $(DOCKERHUB_USER)/$(IMAGE_DEV) $(IMAGE_DEV)

.PHONY: build run test test-mcts-unit test-mcts smoke-phase2 test-mcts-tree-unit test-mcts-tree ablation-phase3 k-ablation-phase3 train eval render push pull upload
