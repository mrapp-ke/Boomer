project('common', 'cpp')

# Source files
source_files = [
    'src/common/binning/bin_index_vector_dense.cpp',
    'src/common/binning/bin_index_vector_dok.cpp',
    'src/common/binning/bin_weight_vector.cpp',
    'src/common/binning/feature_binning_equal_frequency.cpp',
    'src/common/binning/feature_binning_equal_width.cpp',
    'src/common/binning/feature_binning_no.cpp',
    'src/common/binning/threshold_vector.cpp',
    'src/common/data/list_of_lists.cpp',
    'src/common/data/matrix_c_contiguous.cpp',
    'src/common/data/matrix_sparse_set.cpp',
    'src/common/data/ring_buffer.cpp',
    'src/common/data/vector_binned_dense.cpp',
    'src/common/data/vector_bit.cpp',
    'src/common/data/vector_dense.cpp',
    'src/common/data/vector_dok.cpp',
    'src/common/data/vector_dok_binary.cpp',
    'src/common/data/vector_sparse_array.cpp',
    'src/common/data/view_c_contiguous.cpp',
    'src/common/data/view_csc.cpp',
    'src/common/data/view_csc_binary.cpp',
    'src/common/data/view_csr.cpp',
    'src/common/data/view_csr_binary.cpp',
    'src/common/data/view_fortran_contiguous.cpp',
    'src/common/data/view_vector.cpp',
    'src/common/indices/index_iterator.cpp',
    'src/common/indices/index_vector_complete.cpp',
    'src/common/indices/index_vector_partial.cpp',
    'src/common/input/feature_info_equal.cpp',
    'src/common/input/feature_info_mixed.cpp',
    'src/common/input/feature_matrix_c_contiguous.cpp',
    'src/common/input/feature_matrix_csc.cpp',
    'src/common/input/feature_matrix_csr.cpp',
    'src/common/input/feature_matrix_fortran_contiguous.cpp',
    'src/common/input/feature_type_nominal.cpp',
    'src/common/input/feature_type_numerical.cpp',
    'src/common/input/feature_type_ordinal.cpp',
    'src/common/input/feature_vector.cpp',
    'src/common/input/label_matrix_c_contiguous.cpp',
    'src/common/input/label_matrix_csc.cpp',
    'src/common/input/label_matrix_csr.cpp',
    'src/common/input/missing_feature_vector.cpp',
    'src/common/model/body_conjunctive.cpp',
    'src/common/model/body_empty.cpp',
    'src/common/model/condition_list.cpp',
    'src/common/model/head_complete.cpp',
    'src/common/model/head_partial.cpp',
    'src/common/model/rule_list.cpp',
    'src/common/multi_threading/multi_threading_manual.cpp',
    'src/common/multi_threading/multi_threading_no.cpp',
    'src/common/post_optimization/model_builder_intermediate.cpp',
    'src/common/post_optimization/post_optimization_phase_list.cpp',
    'src/common/post_optimization/post_optimization_sequential.cpp',
    'src/common/post_optimization/post_optimization_unused_rule_removal.cpp',
    'src/common/post_processing/post_processor_no.cpp',
    'src/common/prediction/label_space_info_no.cpp',
    'src/common/prediction/label_vector_set.cpp',
    'src/common/prediction/prediction_matrix_dense.cpp',
    'src/common/prediction/prediction_matrix_sparse_binary.cpp',
    'src/common/prediction/probability_calibration_isotonic.cpp',
    'src/common/prediction/probability_calibration_no.cpp',
    'src/common/rule_evaluation/score_vector_binned_dense.cpp',
    'src/common/rule_evaluation/score_vector_dense.cpp',
    'src/common/rule_induction/rule_induction_top_down_beam_search.cpp',
    'src/common/rule_induction/rule_induction_top_down_greedy.cpp',
    'src/common/rule_model_assemblage/default_rule.cpp',
    'src/common/rule_model_assemblage/rule_model_assemblage_sequential.cpp',
    'src/common/rule_pruning/rule_pruning_irep.cpp',
    'src/common/rule_pruning/rule_pruning_no.cpp',
    'src/common/rule_refinement/prediction_complete.cpp',
    'src/common/rule_refinement/prediction_evaluated.cpp',
    'src/common/rule_refinement/prediction_partial.cpp',
    'src/common/rule_refinement/prediction.cpp',
    'src/common/rule_refinement/refinement_comparator_fixed.cpp',
    'src/common/rule_refinement/refinement_comparator_single.cpp',
    'src/common/rule_refinement/rule_refinement_approximate.cpp',
    'src/common/rule_refinement/rule_refinement_exact.cpp',
    'src/common/rule_refinement/score_processor.cpp',
    'src/common/sampling/feature_sampling_no.cpp',
    'src/common/sampling/feature_sampling_predefined.cpp',
    'src/common/sampling/feature_sampling_without_replacement.cpp',
    'src/common/sampling/instance_sampling_no.cpp',
    'src/common/sampling/instance_sampling_stratified_example_wise.cpp',
    'src/common/sampling/instance_sampling_stratified_label_wise.cpp',
    'src/common/sampling/instance_sampling_with_replacement.cpp',
    'src/common/sampling/instance_sampling_without_replacement.cpp',
    'src/common/sampling/label_sampling_no.cpp',
    'src/common/sampling/label_sampling_round_robin.cpp',
    'src/common/sampling/label_sampling_without_replacement.cpp',
    'src/common/sampling/partition_bi.cpp',
    'src/common/sampling/partition_sampling_bi_random.cpp',
    'src/common/sampling/partition_sampling_bi_stratified_example_wise.cpp',
    'src/common/sampling/partition_sampling_bi_stratified_label_wise.cpp',
    'src/common/sampling/partition_sampling_no.cpp',
    'src/common/sampling/partition_single.cpp',
    'src/common/sampling/random.cpp',
    'src/common/sampling/stratified_sampling_example_wise.cpp',
    'src/common/sampling/stratified_sampling_label_wise.cpp',
    'src/common/sampling/weight_vector_bit.cpp',
    'src/common/sampling/weight_vector_dense.cpp',
    'src/common/sampling/weight_vector_equal.cpp',
    'src/common/sampling/weight_vector_out_of_sample.cpp',
    'src/common/stopping/global_pruning_post.cpp',
    'src/common/stopping/global_pruning_pre.cpp',
    'src/common/stopping/stopping_criterion_list.cpp',
    'src/common/stopping/stopping_criterion_size.cpp',
    'src/common/stopping/stopping_criterion_time.cpp',
    'src/common/thresholds/coverage_mask.cpp',
    'src/common/thresholds/coverage_set.cpp',
    'src/common/thresholds/thresholds_approximate.cpp',
    'src/common/thresholds/thresholds_exact.cpp',
    'src/common/learner.cpp'
]

# Directory containing public headers
include_directories = include_directories('include')

# Directory into which the library should be installed
install_root = meson.current_source_dir() / '../../../python/subprojects'
install_dir = install_root / meson.project_name() / 'mlrl' / meson.project_name() / 'cython/'

# Library version
fs = import('fs')
version = fs.read('../../../VERSION')

# Library declaration
lib_name = 'mlrl' + meson.project_name()
cpp_args = ['-fopenmp']
link_args = ['-fopenmp']

if host_machine.system() == 'windows'
    cpp_args = ['/openmp', '-DMLRLCOMMON_EXPORTS']
    link_args = []
elif host_machine.system() == 'darwin'
    cpp_args = ['-Xclang', '-fopenmp']
    link_args = ['-lomp']
endif

library(lib_name, source_files, include_directories : include_directories, cpp_args : cpp_args, link_args : link_args,
        version : version, install : true, install_dir : install_dir)
common_lib = static_library(lib_name, source_files, include_directories : include_directories, cpp_args : cpp_args,
                            link_args : link_args)
common_dep = declare_dependency(include_directories : include_directories, link_with : common_lib)
