PY=python
PYTHONPATH=./src

.PHONY: install run-app run-cli lint test

install:
	$(PY) -m pip install -r requirements.txt

run-app:
	PYTHONPATH=$(PYTHONPATH) streamlit run src/backtester/app.py

run-cli:
	PYTHONPATH=$(PYTHONPATH) $(PY) src/backtester/main.py --help
