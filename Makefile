### Created by Jan - allows externals to easily download our env and requirements
# PYENV_ROOT := $(HOME)/.pyenv
# PYTHON_VERSION := 3.10.6  # Replace with the required Python version
# VENV_NAME := .python-version

# export PATH := $(PYENV_ROOT)/bin:$(PATH)

# .PHONY: install install-pyenv setup-venv activate clean

# install: install-pyenv setup-venv ## Full environment setup

# install-pyenv:
# 	@if ! command -v pyenv >/dev/null; then \
# 		echo "Pyenv is not installed, installing..."; \
# 		curl https://pyenv.run | bash; \
# 		export PATH="$(PYENV_ROOT)/bin:$(PATH)"; \
# 		eval "$$(pyenv init --path)"; \
# 		eval "$$(pyenv virtualenv-init -)"; \
# 	fi

# setup-venv:
# 	@if ! pyenv versions | grep -q $(PYTHON_VERSION); then \
# 		echo "Installing Python $(PYTHON_VERSION) with pyenv..."; \
# 		pyenv install $(PYTHON_VERSION); \
# 	fi
# 	@if [ ! -d "$(VENV_NAME)" ]; then \
# 		echo "Creating virtual environment $(VENV_NAME)..."; \
# 		pyenv virtualenv $(PYTHON_VERSION) $(VENV_NAME); \
# 	fi
# 	echo "Activating environment $(VENV_NAME)..."
# 	pyenv local $(VENV_NAME)
# 	pip install --upgrade pip
# 	@if [ -f requirements.txt ]; then pip install -r requirements.txt; fi

# activate:
# 	@echo "Run: source $(PYENV_ROOT)/versions/$(VENV_NAME)/bin/activate"

# clean:
# 	rm -rf $(VENV_NAME)
