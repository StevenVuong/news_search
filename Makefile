initialise:
	@echo "Initializing git..."
	git init
	@echo "Initializing Poetry"
	poetry init
	poetry env use /bin/python3.10
	
install: 
	@echo "Installing..."
	poetry install
	poetry run pre-commit install

activate:
	@echo "Activating virtual environment"
	poetry env use /bin/python3.10
	poetry shell

# run streamlit app
run_app:
	clear
	@echo "Running streamlit app"
	streamlit run ./app.py

format:
	@echo "Formatting..."
	black ./
	isort ./
	interrogate -vv ./

test:
	pytest

## Delete all compiled Python files
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete
	rm -rf .pytest_cache