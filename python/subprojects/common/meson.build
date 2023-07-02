project('common', 'cython', 'cpp')

# Source files
cython_dir = 'cython'
cython_module_names = [
    'feature_binning',
    'feature_info',
    'feature_matrix',
    'feature_sampling',
    'instance_sampling',
    'label_sampling',
    'label_matrix',
    'label_space_info',
    'learner',
    'multi_threading',
    'partition_sampling',
    'post_optimization',
    'prediction',
    'probability_calibration',
    'rule_induction',
    'rule_model',
    'stopping_criterion'
]

# Dependencies
python = import('python').find_installation()
python_dep = python.dependency()
numpy_cmd = run_command(python.full_path(), '-c', 'import numpy; print(numpy.get_include())', check : true )
numpy_dep = declare_dependency(include_directories : include_directories(numpy_cmd.stdout().strip()))

cpp = meson.get_compiler('cpp')
cpp_dir = '../../../cpp/'
common_lib = cpp.find_library('mlrlcommon', dirs : meson.current_source_dir() / cpp_dir / 'build/subprojects/common/')
common_include = declare_dependency(include_directories : include_directories(cpp_dir / 'subprojects/common/include/'))

dependencies = [
    python_dep,
    numpy_dep,
    common_lib,
    common_include
]

# Cython compiler arguments
cython_options = [
    'cython_language=cpp',
    'cython_version=3'
]

cython_args = [
    '-I', meson.current_source_dir()
]

# C++ compiler arguments

cpp_args = [
    '-DNPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION'
]

# Directory into which the extension modules and Python source files should be installed
python_version = python.language_version()
package_dir = 'mlrl' / meson.project_name()
install_dir = meson.current_source_dir() / package_dir / cython_dir

# Extension modules
rpath = '$ORIGIN'

if host_machine.system() == 'darwin'
    rpath = '@loader_path'
endif

foreach module_name : cython_module_names
    pyx_file = package_dir / cython_dir / module_name + '.pyx'
    python.extension_module(module_name, pyx_file, dependencies : dependencies, override_options : cython_options,
                            cpp_args : cpp_args, install : true, install_dir : install_dir, install_rpath : rpath)
endforeach
