CONFIG=$1
GPU=$2

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
CUDA_VISIBLE_DEVICES=$GPU \
python $(dirname "$0")/train.py $CONFIG --launcher none
