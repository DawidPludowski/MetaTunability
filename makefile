get_coverage: |
	export PYTHONPATH=`pwd` && pytest -vv --cov=meta_tuner --cov-report=term-missing
	rm .coverage