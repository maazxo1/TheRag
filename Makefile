run:
	venv/Scripts/python app.py

ingest:
	venv/Scripts/python entrypoint/ingest.py

test:
	venv/Scripts/python -m pytest tests/unit/ -v

install:
	pip install -r requirements.txt
