.PHONY: build jupyter shell run run-all figures-clean help

CHAPTERS := \
	ch06_linear_regression \
	ch07_logistic_regression \
	ch08_glm_gam \
	ch09_decision_tree \
	ch10_decision_rules \
	ch11_rulefit \
	ch12_ceteris_paribus \
	ch13_ice \
	ch14_lime \
	ch15_counterfactual \
	ch16_anchors \
	ch17_shapley_values \
	ch18_shap \
	ch19_pdp \
	ch20_ale \
	ch21_feature_interaction \
	ch22_functional_decomposition \
	ch23_permutation_importance \
	ch24_lofo \
	ch25_surrogate_models \
	ch26_prototypes \
	ch27_learned_features \
	ch28_saliency_maps \
	ch29_concepts \
	ch30_adversarial_examples \
	ch31_influential_instances

## Build the Docker image
build:
	docker compose build

## Start Jupyter Lab at http://localhost:8888 (no token required)
jupyter:
	@echo "Jupyter Lab starting at http://localhost:8888"
	docker compose up jupyter

## Open an interactive bash shell inside the container
shell:
	docker compose run --rm app bash

## Run a single chapter  (usage: make run CHAPTER=ch06_linear_regression)
run:
	@test -n "$(CHAPTER)" || (echo "Usage: make run CHAPTER=ch06_linear_regression" && exit 1)
	docker compose run --rm app python $(CHAPTER)/main.py

## Run all 26 chapters sequentially
run-all:
	@for ch in $(CHAPTERS); do \
		echo "\n=== $$ch ==="; \
		docker compose run --rm app python $$ch/main.py; \
	done

## Delete all generated figure files (keeps directory structure)
figures-clean:
	find figures -mindepth 2 -type f -delete

## Show available make targets
help:
	@echo "Targets:"
	@echo "  build          Build Docker image"
	@echo "  jupyter        Start Jupyter Lab at http://localhost:8888"
	@echo "  shell          Open shell in container"
	@echo "  run CHAPTER=X  Run one chapter (e.g. CHAPTER=ch06_linear_regression)"
	@echo "  run-all        Run all 26 chapters"
	@echo "  figures-clean  Remove generated figures"
	@echo ""
	@echo "Chapters: $(CHAPTERS)"
