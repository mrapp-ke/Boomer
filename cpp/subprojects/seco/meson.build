project('seco', 'cpp', version : '0.4.0')

# Source files
source_files = [
    'src/seco/head_refinement/head_refinement_partial.cpp',
    'src/seco/head_refinement/lift_function_peak.cpp',
    'src/seco/heuristics/heuristic_f_measure.cpp',
    'src/seco/heuristics/heuristic_hamming_loss.cpp',
    'src/seco/heuristics/heuristic_m_estimate.cpp',
    'src/seco/heuristics/heuristic_precision.cpp',
    'src/seco/heuristics/heuristic_recall.cpp',
    'src/seco/heuristics/heuristic_wra.cpp',
    'src/seco/model/decision_list.cpp',
    'src/seco/output/predictor_classification_label_wise.cpp',
    'src/seco/rule_evaluation/rule_evaluation_label_wise_heuristic.cpp',
    'src/seco/statistics/statistics_label_wise_dense.cpp',
    'src/seco/statistics/statistics_label_wise_provider.cpp',
    'src/seco/stopping/stopping_criterion_coverage.cpp'
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

seco_lib = library('mlrlseco', source_files, include_directories : include_directories, dependencies : dependencies,
                   version : '0.4.0')
