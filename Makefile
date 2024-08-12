lint:
	pylint src

format:
	@echo "Running black..."
	@black .
	@echo "Running isort..."
	@isort .
	@echo "Formatting complete!"

test:
	pytest test
.PHONY: lint format test