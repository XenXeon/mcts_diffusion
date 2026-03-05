IMAGE_BASE = cleandiffuser:base
IMAGE_DEV  = cleandiffuser:dev
WORKDIR    = /workspace

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

.PHONY: build run test
