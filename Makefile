app:
	uv run python src/app/main.py

app-debug:
	uv run chainlit run src/app/main.py -d --port 8001 -w

test:
	uv run pytest -v --cov=src --cov-report=html --cov-report=term

ruff:
	uv run ruff check src --fix --select I

mypy:
	uv run mypy src

GCP_REGION ?= australia-southeast1
GCP_PROJECT_ID ?= gen-ai-466406
GCP_ARTIFACT_REPOSITORY ?= app-images
DOCKER_IMAGE_NAME ?= rag-app

build_docker_image:
	docker image build --no-cache . --tag $(DOCKER_IMAGE_NAME):latest

tag_docker_image_for_gcp:
	docker tag $(DOCKER_IMAGE_NAME):latest $(GCP_REGION)-docker.pkg.dev/$(GCP_PROJECT_ID)/$(GCP_ARTIFACT_REPOSITORY)/$(DOCKER_IMAGE_NAME):latest

push_docker_image_to_gcp:
	docker push $(GCP_REGION)-docker.pkg.dev/$(GCP_PROJECT_ID)/$(GCP_ARTIFACT_REPOSITORY)/$(DOCKER_IMAGE_NAME):latest