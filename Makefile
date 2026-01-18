install:
	python -m pip install -r requirements.txt

lint:
	python -m compileall src

test:
	python -m unittest discover -s tests

run:
	python -m src.run_pipeline