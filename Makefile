default_target: compile
.PHONY: clean_venv clean_compile clean compile install

clean_venv:
	@echo "Removing virtual Python environment..."
	rm -rf venv/

clean_compile:
	@echo "Removing compiled C/C++ files..."
	find python/ -type f -name "*.o" -delete
	find python/ -type f -name "*.so" -delete
	find python/ -type f -name "*.c" -delete
	find python/ -type f -name "*.pyd" -delete
	find python/ -type f -name "*.cpp" -not -path "**/cpp/*" -delete
	find python/ -type f -name "*.html" -delete

clean: clean_compile clean_venv

venv:
	@echo "Creating virtual Python environment..."
	python3 -m venv venv
	@echo "Installing compile-time dependency \"numpy\" into virtual environment..."
	venv/bin/pip install numpy
	@echo "Installing compile-time dependency \"scipy\" into virtual environment..."
	venv/bin/pip install scipy
	@echo "Installing compile-time dependency \"Cython\" into virtual environment..."
	venv/bin/pip install Cython

compile: venv
	@echo "Compiling Cython code..."
	cd python/ && ../venv/bin/python setup.py build_ext --inplace

install: compile
	@echo "Installing package into virtual environment..."
	venv/bin/pip install python/
