# Makefile for EvoLib API maintenance

PYTHON := python3
TOOLS_DIR := tools
SRC_DIR := evolib
API_FILE := $(SRC_DIR)/api.py

.PHONY: all api clean init check-api

all: api

init:
	@echo "Generating __init__.py in all submodules..."
	python tools/gen_init.py evolib
	@echo "Submodule __init__.py files updated."

api: init
	@echo "Generating api.py and evolib/__init__.py ..."
	python tools/gen_api_module.py evolib
	@echo "API ready."

check-api:
	git diff --exit-code evolib/api.py evolib/__init__.py

clean:
	find evolib -name '__pycache__' -type d -exec rm -r {} +
	find evolib -name '*.pyc' -delete

test:
	python -m pytest tests/

coverage:
	pytest --cov=evolib --cov-report=term-missing tests/

docs:
	cd docs && make htm
