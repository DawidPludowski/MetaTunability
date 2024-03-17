CNT_PY_LINES != find ./ -type f -name "*.py" -exec wc -w {} + | tail -n 1
CNT_PY_PACKAGE_LINES != find ./meta_tuner -type f -name "*.py" -exec wc -w {} + | tail -n 1
CNT_PY_TEST_LINES != find ./tests -type f -name "*.py" -exec wc -w {} + | tail -n 1

get_coverage: |
	export PYTHONPATH=`pwd` && pytest -vv --cov=meta_tuner --cov-report=term-missing
	rm .coverage

get_lines: |
	@echo "total python lines (all):$(CNT_PY_LINES)"
	@echo "total python lines (main package):$(CNT_PY_PACKAGE_LINES)"
	@echo "total python lines (tests):$(CNT_PY_TEST_LINES)"