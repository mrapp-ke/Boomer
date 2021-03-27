default_target: compile
.PHONY: clean_venv clean_cpp clean_cython clean_compile clean install

clean_venv:
	@echo "Removing virtual Python environment..."
	rm -rf venv/

clean_cpp:
	@echo "Removing C++ compilation files..."
	rm -rf cpp/build/

clean_cython:
	@echo "Removing Cython compilation files..."
	find python/ -type f -name "*.o" -delete
	find python/ -type f -name "*.so" -delete
	find python/ -type f -name "*.c" -delete
	find python/ -type f -name "*.cpp" -delete
	find python/ -type f -name "*.pyd" -delete
	find python/ -type f -name "*.pyc" -delete
	find python/ -type f -name "*.html" -delete

clean_compile: clean_cpp clean_cython

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
	@echo "Installing compile-time dependency \"meson\" into virtual environment..."
	venv/bin/pip install meson
	@echo "Installing compile-time dependency \"ninja\" into virtual environment..."
	venv/bin/pip install ninja
	@echo "Installing compile-time dependency \"wheel\" into virtual environment..."
	venv/bin/pip install wheel

compile: venv
	@echo "Compiling C++ code..."
	cd cpp/ && PATH=$$PATH:../venv/bin/ ../venv/bin/meson setup build -Doptimization=3 && cd build/ && ../../venv/bin/ninja
	@echo "Compiling Cython code..."
	cd python/ && ../venv/bin/python setup.py build_ext --inplace

install: compile
	@echo "Installing package into virtual environment..."
	venv/bin/pip install python/
