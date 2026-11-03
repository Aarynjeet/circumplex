# Common tasks. Every target is a thin wrapper around a python module or script.
PY ?= python

.PHONY: install smoke test test-fast lint format baseline tune-baseline vad analysis train-bilstm train-full train-lora transfer embeddings attributions demo clean-smoke

install:            ## install pinned dependencies + this package (editable)
	$(PY) -m pip install -r requirements.txt
	$(PY) -m pip install -e .

smoke:              ## CPU smoke tests of every training path (about 1 min total, offline after first download)
	$(PY) -m circumplex.train --config configs/smoke.yaml
	$(PY) -m circumplex.train --config configs/smoke_transformer.yaml
	$(PY) -m circumplex.train --config configs/smoke_lora.yaml
	$(PY) -m circumplex.transfer --config configs/smoke_transfer.yaml

test:               ## full unit-test suite
	$(PY) -m pytest

test-fast:          ## unit tests without the ones that train on downloaded data
	$(PY) -m pytest -m "not slow"

lint:               ## ruff lint + format check
	ruff check src tests scripts demo notebooks
	ruff format --check src tests scripts demo

format:
	ruff format src tests scripts demo
	ruff check --fix src tests scripts demo

baseline:           ## TF-IDF + logistic regression (CPU, about 1 min)
	$(PY) -m circumplex.baselines --config configs/tfidf_lr.yaml

tune-baseline:      ## dev-set grid search for the baseline's C / class_weight (about 10 min)
	$(PY) scripts/tune_tfidf.py

vad:                ## build data/emotion_vad.csv from the official NRC VAD lexicon (needs internet)
	$(PY) scripts/build_vad.py

leakage:            ## train/test text overlap + deduplicated test scores for every run
	$(PY) scripts/check_leakage.py

analysis:           ## regenerate every figure + runs/analysis/analysis.json from runs/
	$(PY) scripts/run_analysis.py

train-bilstm:
	$(PY) -m circumplex.train --config configs/bilstm.yaml

train-full:
	$(PY) -m circumplex.train --config configs/transformer_full.yaml --set name=roberta_full_s42 seed=42

train-lora:
	$(PY) -m circumplex.train --config configs/transformer_lora.yaml --set name=roberta_lora_s42 seed=42

transfer:
	$(PY) -m circumplex.transfer --config configs/transfer.yaml

embeddings:
	$(PY) scripts/extract_embeddings.py --run runs/roberta_full_s42

attributions:
	$(PY) scripts/attributions.py --run runs/roberta_full_s42 --out figures/attributions.html

demo:
	$(PY) demo/app.py

clean-smoke:
	rm -rf runs/smoke_*
