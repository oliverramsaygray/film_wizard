PYENV_ROOT := $(HOME)/.pyenv
PYTHON_VERSION := 3.10.6  # Replace with the required Python version
VENV_NAME := film_wizard  # Name of the virtual environment

export PATH := $(PYENV_ROOT)/bin:$(PATH)

.PHONY: install install-pyenv setup-venv activate clean install-requirements

install: install-pyenv setup-venv install-requirements ## Full environment setup

install-pyenv:
	@if ! command -v pyenv >/dev/null; then \
		echo "Pyenv is not installed, installing..."; \
		curl https://pyenv.run | bash; \
		export PATH="$(PYENV_ROOT)/bin:$(PATH)"; \
		eval "$(shell pyenv init --path)"; \
		eval "$(shell pyenv virtualenv-init -)"; \
	fi

setup-venv:
	@if ! pyenv versions | grep -q $(PYTHON_VERSION); then \
		echo "Installing Python $(PYTHON_VERSION) with pyenv..."; \
		pyenv install $(PYTHON_VERSION); \
	fi
	@if ! pyenv virtualenvs | grep -q $(VENV_NAME); then \
		echo "Creating virtual environment $(VENV_NAME)..."; \
		pyenv virtualenv $(PYTHON_VERSION) $(VENV_NAME); \
	fi
	echo "Setting local environment to $(VENV_NAME)..."
	pyenv local $(VENV_NAME)
	pip install --upgrade pip

install-requirements:
	@echo "Installing required packages from requirements.txt..."
	@if [ -f requirements.txt ]; then pip install -r requirements.txt; fi
	@echo "Installation completed."

activate:
	@echo "Run the following command to activate the environment:"
	@echo "source $(PYENV_ROOT)/versions/$(VENV_NAME)/bin/activate"

clean:
	rm -rf $(PYENV_ROOT)/versions/$(VENV_NAME)
