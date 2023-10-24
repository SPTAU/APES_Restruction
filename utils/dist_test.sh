CONFIG=$1
CHECKPOINT=$2
GPUS=$3
VIS=$4

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

IFS=',' read -ra GPU <<< "$GPUS"
NUM_GPUS=${#GPU[@]}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
CUDA_VISIBLE_DEVICES=$GPUS \
torchrun \
    --standalone \
    --nnodes=1 \
    --nproc_per_node=$NUM_GPUS \
    $(dirname "$0")/test.py \
    $CONFIG \
    $CHECKPOINT \
    --launcher pytorch \
    $VIS
