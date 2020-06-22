#!/bin/bash

# Command line arguments
DATASET=$1
LOSS=$2

# Constants
FOLDS=10
INSTANCE_SUB_SAMPLING="bagging"
FEATURE_SUB_SAMPLING="random-feature-selection"
NUM_RULES=10000
TIME_LIMIT=82800 # 23h in seconds

# Grid search parameters
HEAD_REFINEMENT_ARRAY=("single-label" "full")
LABEL_SUB_SAMPLING_ARRAY=(-1)
SHRINKAGE_ARRAY=(0.1)
L2_REGULARIZATION_WEIGHT_ARRAY=(0.0 0.25 1.0 4.0 16.0 64.0)

# Paths
ROOT_DIR="${PWD}"
WORK_DIR="${ROOT_DIR}/results/${LOSS}/${DATASET}"
LOG_DIR="${WORK_DIR}/logs"
MODEL_DIR="${ROOT_DIR}/models/${LOSS}/${DATASET}"
DATA_DIR="${ROOT_DIR}/data"

MEMORY=4096

# Create directories
echo "Creating directory ${LOG_DIR}"
mkdir -p "${LOG_DIR}"
echo "Creating directory ${MODEL_DIR}"
mkdir -p "${MODEL_DIR}"

COUNT=1

for SHRINKAGE in "${SHRINKAGE_ARRAY[@]}"
do
  for L2_REGULARIZATION_WEIGHT in "${L2_REGULARIZATION_WEIGHT_ARRAY[@]}"
  do
    for HEAD_REFINEMENT in "${HEAD_REFINEMENT_ARRAY[@]}"
    do
      for LABEL_SUB_SAMPLING in "${LABEL_SUB_SAMPLING_ARRAY[@]}"
      do
        if [ "$HEAD_REFINEMENT" = "full" ] || [ $LABEL_SUB_SAMPLING -eq -1 ]
        then
          for CURRENT_FOLD in $(seq 1 $FOLDS)
          do
            JOB_NAME="${DATASET}-${CURRENT_FOLD}_loss=${LOSS}_${COUNT}"
            FILE="${JOB_NAME}.sh"
            PARAMETERS="--log-level error --data-dir ${DATA_DIR} --dataset ${DATASET} --model-dir ${MODEL_DIR} --folds ${FOLDS} --current-fold ${CURRENT_FOLD} --instance-sub-sampling ${INSTANCE_SUB_SAMPLING} --feature-sub-sampling ${FEATURE_SUB_SAMPLING} --num-rules ${NUM_RULES} --time-limit ${TIME_LIMIT} --shrinkage ${SHRINKAGE} --loss ${LOSS} --head-refinement ${HEAD_REFINEMENT} --label-sub-sampling ${LABEL_SUB_SAMPLING} --l2-regularization-weight ${L2_REGULARIZATION_WEIGHT}"

            echo "$FILE"
            echo "#!/bin/sh" >> "$FILE"
            echo "#SBATCH -J ${JOB_NAME}" >> "$FILE"
            echo "#SBATCH -D ${WORK_DIR}" >> "$FILE"
            echo "#SBATCH -t 24:00:00" >> "$FILE"
            echo "#SBATCH -c 1" >> "$FILE"
            echo "#SBATCH -o /dev/null" >> "$FILE"
            echo "#SBATCH -e logs/%J.log" >> "$FILE"
            echo "#SBATCH --mem-per-cpu=${MEMORY}" >> "$FILE"
            echo "#SBATCH --ntasks=1" >> "$FILE"
            echo "${ROOT_DIR}/venv/bin/python3.7 ${ROOT_DIR}/python/main_boomer.py ${PARAMETERS}" >> "$FILE"
            chmod +x "$FILE"

            # Run SLURM job
            sbatch "$FILE"
            rm "$FILE"
            echo "Started experiment with parameters ${PARAMETERS}"
          done

          COUNT=$((COUNT + 1))
        fi
      done
    done
  done
done
