project('common', 'cpp', version : '0.4.0')

# Source files
source_files = [
    'src/common/binning/bin_index_vector_dense.cpp',
    'src/common/binning/bin_index_vector_dok.cpp',
    'src/common/binning/feature_binning_equal_frequency.cpp',
    'src/common/binning/feature_binning_equal_width.cpp',
    'src/common/binning/feature_binning_nominal.cpp',
    'src/common/binning/threshold_vector.cpp',
    'src/common/data/matrix_dense.cpp',
    'src/common/data/matrix_dok_binary.cpp',
    'src/common/data/ring_buffer.cpp',
    'src/common/data/vector_dense.cpp',
    'src/common/data/vector_dok.cpp',
    'src/common/data/vector_dok_binary.cpp',
    'src/common/data/vector_sparse_array.cpp',
    'src/common/data/vector_sparse_list_binary.cpp',
    'src/common/data/view_c_contiguous.cpp',
    'src/common/data/view_csc.cpp',
    'src/common/data/view_csr.cpp',
    'src/common/data/view_fortran_contiguous.cpp',
    'src/common/head_refinement/head_refinement_full.cpp',
    'src/common/head_refinement/head_refinement_single.cpp',
    'src/common/head_refinement/prediction_evaluated.cpp',
    'src/common/head_refinement/prediction_full.cpp',
    'src/common/head_refinement/prediction_partial.cpp',
    'src/common/head_refinement/prediction.cpp',
    'src/common/indices/index_iterator.cpp',
    'src/common/indices/index_vector_full.cpp',
    'src/common/indices/index_vector_partial.cpp',
    'src/common/input/feature_matrix_csc.cpp',
    'src/common/input/feature_matrix_fortran_contiguous.cpp',
    'src/common/input/feature_vector.cpp',
    'src/common/input/label_matrix_c_contiguous.cpp',
    'src/common/input/label_matrix_dok.cpp',
    'src/common/input/missing_feature_vector.cpp',
    'src/common/input/nominal_feature_mask_dok.cpp',
    'src/common/input/nominal_feature_mask_equal.cpp',
    'src/common/model/body_conjunctive.cpp',
    'src/common/model/body_empty.cpp',
    'src/common/model/condition_list.cpp',
    'src/common/model/condition.cpp',
    'src/common/model/head_full.cpp',
    'src/common/model/head_partial.cpp',
    'src/common/model/rule_model.cpp',
    'src/common/model/rule.cpp',
    'src/common/post_processing/post_processor_no.cpp',
    'src/common/pruning/pruning_irep.cpp',
    'src/common/pruning/pruning_no.cpp',
    'src/common/rule_evaluation/score_vector_dense.cpp',
    'src/common/rule_evaluation/score_vector_label_wise_dense.cpp',
    'src/common/rule_induction/rule_induction_top_down.cpp',
    'src/common/rule_induction/rule_model_induction_sequential.cpp',
    'src/common/rule_refinement/refinement.cpp',
    'src/common/rule_refinement/rule_refinement_approximate.cpp',
    'src/common/rule_refinement/rule_refinement_exact.cpp',
    'src/common/sampling/feature_sampling_no.cpp',
    'src/common/sampling/feature_sampling_random.cpp',
    'src/common/sampling/instance_sampling_bagging.cpp',
    'src/common/sampling/instance_sampling_no.cpp',
    'src/common/sampling/instance_sampling_random.cpp',
    'src/common/sampling/label_sampling_no.cpp',
    'src/common/sampling/label_sampling_random.cpp',
    'src/common/sampling/partition_bi.cpp',
    'src/common/sampling/partition_sampling_bi.cpp',
    'src/common/sampling/partition_sampling_no.cpp',
    'src/common/sampling/partition_single.cpp',
    'src/common/sampling/random.cpp',
    'src/common/sampling/weight_vector_dense.cpp',
    'src/common/sampling/weight_vector_equal.cpp',
    'src/common/stopping/stopping_criterion_measure.cpp',
    'src/common/stopping/stopping_criterion_size.cpp',
    'src/common/stopping/stopping_criterion_time.cpp',
    'src/common/thresholds/coverage_mask.cpp',
    'src/common/thresholds/coverage_set.cpp',
    'src/common/thresholds/thresholds_approximate.cpp',
    'src/common/thresholds/thresholds_exact.cpp'
]

# Dependencies
openmp_dep = dependency('openmp')

dependencies = [
    openmp_dep
]

# Directory containing public headers
include_directories = include_directories('include')

common_lib = library('mlrlcommon', source_files, include_directories : include_directories, dependencies : dependencies,
                     version : '0.4.0')
common_dep = declare_dependency(include_directories : include_directories, link_with : common_lib)
