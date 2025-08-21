app:
	uv run python src/chat_my_doc_app/app.py

app-debug:
	uv run python src/chat_my_doc_app/app.py --debug

data:
	uv run python src/chat_my_doc_app/data.py process-imdb-data

unit-test:
	uv run pytest tests/unit -v --cov=src --cov-report=html --cov-report=term

integration-test:
	uv run pytest tests/integration -v --cov=src --cov-report=html --slow

test:
	uv run pytest tests -v --cov=src --cov-report=html --cov-report=term --slow

ruff:
	uv run ruff check . --fix --select I

mypy:
	uv run mypy src

pre-commit-install:
	uv run pre-commit install

pre-commit: pre-commit-install
	uv run pre-commit run --all-files

clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache .coverage htmlcov
	rm -rf tests/unit/__pycache__ tests/integration/__pycache__

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
