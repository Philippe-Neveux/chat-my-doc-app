app:
	uv run python src/chat_my_doc_app/main.py

app-debug:
	uv run python src/chat_my_doc_app/main.py --debug

data:
	uv run python src/chat_my_doc_app/data.py process-imdb-data

test:
	uv run pytest -v --cov=src --cov-report=html --cov-report=term

ruff:
	uv run ruff check src --fix --select I

mypy:
	uv run mypy src

pre-commit-install:
	uv run pre-commit install

pre-commit: pre-commit-install
	uv run pre-commit run --all-files

GCP_REGION ?= australia-southeast1
GCP_PROJECT_ID ?= gen-ai-466406
GCP_ARTIFACT_REPOSITORY ?= app-images
DOCKER_IMAGE_NAME ?= chat-my-doc-app

build_docker_image:
	docker image build --no-cache . --tag $(DOCKER_IMAGE_NAME):latest

tag_docker_image_for_gcp:
	docker tag $(DOCKER_IMAGE_NAME):latest $(GCP_REGION)-docker.pkg.dev/$(GCP_PROJECT_ID)/$(GCP_ARTIFACT_REPOSITORY)/$(DOCKER_IMAGE_NAME):latest

push_docker_image_to_gcp:
	docker push $(GCP_REGION)-docker.pkg.dev/$(GCP_PROJECT_ID)/$(GCP_ARTIFACT_REPOSITORY)/$(DOCKER_IMAGE_NAME):latest
