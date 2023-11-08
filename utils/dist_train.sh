CONFIG=$1
GPUS=$2
VIS=$3

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
    $(dirname "$0")/train.py \
    $CONFIG \
    --launcher pytorch \
    $VIS
