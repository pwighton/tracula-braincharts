IMAGE   := pwighton/tracula-braincharts
VERSION := $(shell sed -n 's/^version *= *"\([^"]*\)".*/\1/p' pyproject.toml)
TAG     := $(IMAGE):$(VERSION)
GIT_SHA := $(shell git rev-parse --short=12 HEAD 2>/dev/null || echo unknown)$(shell git diff --quiet 2>/dev/null || echo -dirty)

.PHONY: build test push version

build:
	docker build --build-arg GIT_SHA=$(GIT_SHA) --build-arg IMAGE_TAG=$(TAG) -t $(TAG) -t $(IMAGE):latest .

test: build
	docker run --rm --user $$(id -u):$$(id -g) --entrypoint pytest $(TAG) -q -p no:cacheprovider

push: build
	docker push $(TAG)
	docker push $(IMAGE):latest

version:
	@echo $(VERSION)
