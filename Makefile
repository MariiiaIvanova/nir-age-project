export HOST_UID := $(shell id -u)
export HOST_GID := $(shell id -g)
RUN = docker compose run --rm
.PHONY: build fixed prepare weights smoke train resume pack dirs
dirs:
	mkdir -p weights output
build: dirs
	docker compose build prepare
fixed:
	@if [ ! -s benchmark/all_train.tsv.gz ]; then \
		set -eu; . ./.env; python3 tools/import_train.py "$$LEGACY_SPLITS/all_train.txt" benchmark/all_train.tsv.gz; \
	fi
prepare: fixed dirs
	$(RUN) prepare python prepare.py $(ARGS)
weights: dirs
	$(RUN) prepare python -c "import whisper; whisper.load_model('tiny', device='cpu', download_root='/weights')"
smoke:
	$(RUN) train python train.py --smoke $(ARGS)
train:
	$(RUN) train python train.py $(ARGS)
resume:
	$(RUN) train python train.py --resume $(ARGS)
pack: fixed
	python3 tools/pack.py
.PHONY: prepare-native
prepare-native: fixed
	@set -a; . ./.env; set +a; python app/prepare.py --host-roots --fixed benchmark --work output/native $(ARGS)

# Direct Python entry points (activate the project environment first).
PYTHON ?= python
NATIVE_SPLITS ?= output/native/validation_seed42
NATIVE_WEIGHTS ?= weights/tiny.pt
NATIVE_RUN ?= output/native/runs/whisper_tiny_adamw_val_seed42
.PHONY: split-native smoke-native train-native resume-native check
split-native:
	$(PYTHON) app/split_validation.py --source output/native/current --dest $(NATIVE_SPLITS) --seed 42
smoke-native:
	$(PYTHON) -u app/train.py --splits $(NATIVE_SPLITS) --weights $(NATIVE_WEIGHTS) --smoke $(ARGS)
train-native:
	$(PYTHON) -u app/train.py --splits $(NATIVE_SPLITS) --weights $(NATIVE_WEIGHTS) --run $(NATIVE_RUN) $(ARGS)
resume-native:
	$(PYTHON) -u app/train.py --splits $(NATIVE_SPLITS) --weights $(NATIVE_WEIGHTS) --run $(NATIVE_RUN) --resume $(ARGS)
check:
	$(PYTHON) tests/check_pipeline.py

.PHONY: weights-native
weights-native:
	$(PYTHON) tools/download_weights.py $(ARGS)
