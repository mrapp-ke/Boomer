#!/bin/bash

# Command line arguments
DATASET=$1
TRANSFORMATION_METHOD=$2

# Constants
FOLDS=10

# Grid search parameters
LEARNING_RATE_ARRAY=(0.1)
REG_LAMBDA_ARRAY=(0.0 0.25 1.0 4.0 16.0 64.0)
CHAIN_ORDER_ARRAY=(1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20)

# Paths
ROOT_DIR="${PWD}"
WORK_DIR="${ROOT_DIR}/results/xgboost/${TRANSFORMATION_METHOD}/${DATASET}"
LOG_DIR="${WORK_DIR}/logs"
MODEL_DIR="${ROOT_DIR}/models/xgboost/${TRANSFORMATION_METHOD}/${DATASET}"
DATA_DIR="${ROOT_DIR}/data"

MEMORY=4096

# Create directories
echo "Creating directory ${LOG_DIR}"
mkdir -p "${LOG_DIR}"
echo "Creating directory ${MODEL_DIR}"
mkdir -p "${MODEL_DIR}"

COUNT=1

for LEARNING_RATE in "${LEARNING_RATE_ARRAY[@]}"
do
  for REG_LAMBDA in "${REG_LAMBDA_ARRAY[@]}"
  do
    for CHAIN_ORDER in "${CHAIN_ORDER_ARRAY[@]}"
    do
      if [ "$TRANSFORMATION_METHOD" = "cc" ] || [ $CHAIN_ORDER -eq -1 ]
      then
        for CURRENT_FOLD in $(seq 1 $FOLDS)
        do
          JOB_NAME="${DATASET}-${CURRENT_FOLD}_xgboost_${TRANSFORMATION_METHOD}_${COUNT}"
          FILE="${JOB_NAME}.sh"
          PARAMETERS="--log-level error --data-dir ${DATA_DIR} --dataset ${DATASET} --model-dir ${MODEL_DIR} --folds ${FOLDS} --current-fold ${CURRENT_FOLD} --learning-rate ${LEARNING_RATE} --reg-lambda ${REG_LAMBDA} --transformation-method ${TRANSFORMATION_METHOD} --chain-order ${CHAIN_ORDER}"

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
          echo "${ROOT_DIR}/venv/bin/python3.7 ${ROOT_DIR}/python/main_xgboost.py ${PARAMETERS}" >> "$FILE"
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
