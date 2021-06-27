project('boosting', 'cpp', version : '0.5.0')

# Source files
source_files = [
    'src/boosting/binning/label_binning_equal_width.cpp',
    'src/boosting/data/matrix_dense_example_wise.cpp',
    'src/boosting/data/matrix_dense_label_wise.cpp',
    'src/boosting/data/matrix_dense_numeric.cpp',
    'src/boosting/data/vector_dense_example_wise.cpp',
    'src/boosting/data/vector_dense_label_wise.cpp',
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
    'src/boosting/rule_evaluation/rule_evaluation_example_wise_binning.cpp',
    'src/boosting/rule_evaluation/rule_evaluation_example_wise_regularized.cpp',
    'src/boosting/rule_evaluation/rule_evaluation_label_wise_binning.cpp',
    'src/boosting/rule_evaluation/rule_evaluation_label_wise_regularized.cpp',
    'src/boosting/statistics/statistics_example_wise_dense.cpp',
    'src/boosting/statistics/statistics_example_wise_provider.cpp',
    'src/boosting/statistics/statistics_label_wise_dense.cpp',
    'src/boosting/statistics/statistics_label_wise_provider.cpp'
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
                       dependencies : dependencies, version : '0.5.0')
