.PHONY: install train test lint all

install:
	pip install -r requirements.txt

train:
	python src/train.py

test:
	pytest tests/ -v

lint:
	flake8 src/ tests/ --max-line-length=120

all: install lint train test
