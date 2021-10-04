project('boosting', 'cpp', version : '0.6.2')

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
common_dep = subproject('common').get_variable('common_dep')
openmp_dep = dependency('openmp')

dependencies = [
    common_dep,
    openmp_dep
]

# Directory containing public headers
include_directories = include_directories('include')

boosting_lib = library('mlrlboosting', source_files, include_directories : include_directories,
                       dependencies : dependencies, version : '0.6.2')
