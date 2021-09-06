default_target: compile
.PHONY: clean_venv clean_cpp clean_cython clean_compile clean_doc clean install doc

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

clean_doc:
	@echo "Removing documentation..."
	rm -rf doc/_build/
	rm -rf doc/doxygen/
	rm -rf doc/python_apidoc/
	rm -f doc/python/*.rst

clean: clean_doc clean_compile clean_venv

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
	cd cpp/ && PATH=$$PATH:../venv/bin/ ../venv/bin/meson setup build/ -Doptimization=3
	cd cpp/build/ && PATH=$$PATH:../../venv/bin/ ../../venv/bin/meson compile
	@echo "Compiling Cython code..."
	cd python/ && ../venv/bin/python setup.py build_ext --inplace

install: compile
	@echo "Installing package into virtual environment..."
	venv/bin/pip install python/

doc: install
	@echo "Installing dependency \"Sphinx\" into virtual environment..."
	venv/bin/pip install Sphinx
	@echo "Installing dependency \"sphinx_rtd_theme\" into virtual environment..."
	venv/bin/pip install sphinx_rtd_theme
	@echo "Generating C++ API documentation via Doxygen..."
	cd doc/ && mkdir -p doxygen/api/cpp/ && doxygen Doxyfile
	@echo "Generating Python API documentation via sphinx-apidoc..."
	venv/bin/sphinx-apidoc --tocfile index -f -o doc/python python/mlrl **/seco **/cython
	cd doc/python/ && PATH=$$PATH:../../venv/bin/ LD_PRELOAD="../../cpp/build/subprojects/common/libmlrlcommon.so \
	    ../../cpp/build/subprojects/boosting/libmlrlboosting.so" sphinx-build -M html . ../python_apidoc/api/python
	@echo "Generating Sphinx documentation..."
	cd doc/ && PATH=$$PATH:../venv/bin/ sphinx-build -M html . build_
