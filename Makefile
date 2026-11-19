.PHONY: dev kb-preview kb-build kb-mine kb-vectorize kb-reset eval-retrieval

dev:
	.venv/Scripts/python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload

kb-preview:
	.venv/Scripts/python.exe -m app.jobs.ingest_docs --preview

kb-build:
	.venv/Scripts/python.exe -m app.jobs.ingest_docs --force

kb-mine:
	.venv/Scripts/python.exe -m app.jobs.qa_mining

kb-vectorize:
	.venv/Scripts/python.exe -m app.jobs.ingest_docs --reconcile

kb-reset:
	.venv/Scripts/python.exe -m app.jobs.ingest_docs --rebuild

eval-retrieval:
	.venv/Scripts/python.exe -m evaluation.run_eval
