project('boosting', 'cpp')

# Source files
source_files = [
    'src/boosting/binning/label_binning_equal_width.cpp',
    'src/boosting/data/matrix_dense_numeric.cpp',
    'src/boosting/data/statistic_vector_example_wise_dense.cpp',
    'src/boosting/data/statistic_vector_label_wise_dense.cpp',
    'src/boosting/data/statistic_view_example_wise_dense.cpp',
    'src/boosting/data/statistic_view_label_wise_dense.cpp',
    'src/boosting/iterator/diagonal_iterator.cpp',
    'src/boosting/losses/loss_example_wise_logistic.cpp',
    'src/boosting/losses/loss_label_wise.cpp',
    'src/boosting/losses/loss_label_wise_logistic.cpp',
    'src/boosting/losses/loss_label_wise_squared_error.cpp',
    'src/boosting/losses/loss_label_wise_squared_hinge.cpp',
    'src/boosting/math/blas.cpp',
    'src/boosting/math/lapack.cpp',
    'src/boosting/model/rule_list.cpp',
    'src/boosting/output/predictor_classification_example_wise.cpp',
    'src/boosting/output/predictor_classification_label_wise.cpp',
    'src/boosting/output/predictor_probability_label_wise.cpp',
    'src/boosting/output/predictor_regression_label_wise.cpp',
    'src/boosting/post_processing/shrinkage_constant.cpp',
    'src/boosting/rule_evaluation/rule_evaluation_example_wise_complete.cpp',
    'src/boosting/rule_evaluation/rule_evaluation_example_wise_complete_binned.cpp',
    'src/boosting/rule_evaluation/rule_evaluation_label_wise_complete.cpp',
    'src/boosting/rule_evaluation/rule_evaluation_label_wise_complete_binned.cpp',
    'src/boosting/rule_evaluation/rule_evaluation_label_wise_single.cpp',
    'src/boosting/statistics/statistics_provider_factory_example_wise_dense.cpp',
    'src/boosting/statistics/statistics_provider_factory_label_wise_dense.cpp'
]

# Dependencies
common_project = subproject('common')
common_dep = common_project.get_variable('common_dep')

dependencies = [
    common_dep
]

# Directory containing public headers
include_directories = include_directories('include')

# Directory into which the library should be installed
install_root = common_project.get_variable('install_root')
install_dir = install_root / meson.project_name() / 'mlrl' / meson.project_name() / 'cython/'

# Library version
version = common_project.get_variable('version')

# Library declaration
cpp_args = common_project.get_variable('cpp_args')
link_args = common_project.get_variable('link_args')
boosting_lib = library('mlrl' + meson.project_name(), source_files, include_directories : include_directories,
                       dependencies : dependencies, cpp_args : cpp_args, link_args : link_args, version : version,
                       install : true, install_dir : install_dir)
