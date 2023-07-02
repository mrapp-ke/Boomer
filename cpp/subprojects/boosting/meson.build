project('boosting', 'cpp')

# Source files
source_files = [
    'src/boosting/binning/feature_binning_auto.cpp',
    'src/boosting/binning/label_binning_auto.cpp',
    'src/boosting/binning/label_binning_equal_width.cpp',
    'src/boosting/binning/label_binning_no.cpp',
    'src/boosting/data/histogram_view_label_wise_sparse.cpp',
    'src/boosting/data/matrix_c_contiguous_numeric.cpp',
    'src/boosting/data/matrix_sparse_set_numeric.cpp',
    'src/boosting/data/statistic_vector_example_wise_dense.cpp',
    'src/boosting/data/statistic_vector_label_wise_dense.cpp',
    'src/boosting/data/statistic_vector_label_wise_sparse.cpp',
    'src/boosting/data/statistic_view_example_wise_dense.cpp',
    'src/boosting/data/statistic_view_label_wise_dense.cpp',
    'src/boosting/data/statistic_view_label_wise_sparse.cpp',
    'src/boosting/iterator/diagonal_iterator.cpp',
    'src/boosting/losses/loss_example_wise_logistic.cpp',
    'src/boosting/losses/loss_example_wise_squared_error.cpp',
    'src/boosting/losses/loss_example_wise_squared_hinge.cpp',
    'src/boosting/losses/loss_label_wise_logistic.cpp',
    'src/boosting/losses/loss_label_wise_squared_error.cpp',
    'src/boosting/losses/loss_label_wise_squared_hinge.cpp',
    'src/boosting/math/blas.cpp',
    'src/boosting/math/lapack.cpp',
    'src/boosting/model/rule_list_builder.cpp',
    'src/boosting/multi_threading/parallel_rule_refinement_auto.cpp',
    'src/boosting/multi_threading/parallel_statistic_update_auto.cpp',
    'src/boosting/post_processing/shrinkage_constant.cpp',
    'src/boosting/prediction/discretization_function_probability.cpp',
    'src/boosting/prediction/discretization_function_score.cpp',
    'src/boosting/prediction/predictor_binary_auto.cpp',
    'src/boosting/prediction/predictor_binary_example_wise.cpp',
    'src/boosting/prediction/predictor_binary_gfm.cpp',
    'src/boosting/prediction/predictor_binary_label_wise.cpp',
    'src/boosting/prediction/predictor_probability_auto.cpp',
    'src/boosting/prediction/predictor_probability_label_wise.cpp',
    'src/boosting/prediction/predictor_probability_marginalized.cpp',
    'src/boosting/prediction/predictor_score_label_wise.cpp',
    'src/boosting/prediction/probability_calibration_isotonic.cpp',
    'src/boosting/prediction/probability_function_chain_rule.cpp',
    'src/boosting/prediction/probability_function_logistic.cpp',
    'src/boosting/prediction/transformation_binary_example_wise.cpp',
    'src/boosting/prediction/transformation_binary_gfm.cpp',
    'src/boosting/prediction/transformation_binary_label_wise.cpp',
    'src/boosting/prediction/transformation_probability_label_wise.cpp',
    'src/boosting/prediction/transformation_probability_marginalized.cpp',
    'src/boosting/rule_evaluation/head_type_auto.cpp',
    'src/boosting/rule_evaluation/head_type_complete.cpp',
    'src/boosting/rule_evaluation/head_type_partial_dynamic.cpp',
    'src/boosting/rule_evaluation/head_type_partial_fixed.cpp',
    'src/boosting/rule_evaluation/head_type_single.cpp',
    'src/boosting/rule_evaluation/regularization_manual.cpp',
    'src/boosting/rule_evaluation/regularization_no.cpp',
    'src/boosting/rule_evaluation/rule_evaluation_example_wise_complete.cpp',
    'src/boosting/rule_evaluation/rule_evaluation_example_wise_complete_binned.cpp',
    'src/boosting/rule_evaluation/rule_evaluation_example_wise_partial_dynamic.cpp',
    'src/boosting/rule_evaluation/rule_evaluation_example_wise_partial_dynamic_binned.cpp',
    'src/boosting/rule_evaluation/rule_evaluation_example_wise_partial_fixed.cpp',
    'src/boosting/rule_evaluation/rule_evaluation_example_wise_partial_fixed_binned.cpp',
    'src/boosting/rule_evaluation/rule_evaluation_label_wise_complete.cpp',
    'src/boosting/rule_evaluation/rule_evaluation_label_wise_complete_binned.cpp',
    'src/boosting/rule_evaluation/rule_evaluation_label_wise_partial_dynamic.cpp',
    'src/boosting/rule_evaluation/rule_evaluation_label_wise_partial_dynamic_binned.cpp',
    'src/boosting/rule_evaluation/rule_evaluation_label_wise_partial_fixed.cpp',
    'src/boosting/rule_evaluation/rule_evaluation_label_wise_partial_fixed_binned.cpp',
    'src/boosting/rule_evaluation/rule_evaluation_label_wise_single.cpp',
    'src/boosting/rule_model_assemblage/default_rule_auto.cpp',
    'src/boosting/sampling/partition_sampling_auto.cpp',
    'src/boosting/statistics/statistic_format_auto.cpp',
    'src/boosting/statistics/statistic_format_dense.cpp',
    'src/boosting/statistics/statistic_format_sparse.cpp',
    'src/boosting/statistics/statistics_provider_example_wise_dense.cpp',
    'src/boosting/statistics/statistics_provider_label_wise_dense.cpp',
    'src/boosting/statistics/statistics_provider_label_wise_sparse.cpp',
    'src/boosting/learner.cpp',
    'src/boosting/learner_boomer.cpp'
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

if host_machine.system() == 'windows'
    cpp_args += '-DMLRLBOOSTING_EXPORTS'
endif

boosting_lib = library('mlrl' + meson.project_name(), source_files, include_directories : include_directories,
                       dependencies : dependencies, cpp_args : cpp_args, link_args : link_args, version : version,
                       install : true, install_dir : install_dir)
