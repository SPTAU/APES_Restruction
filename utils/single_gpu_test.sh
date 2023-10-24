CONFIG=$1
CHECKPOINT=$2
GPU=$3
VIS=$4

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
CUDA_VISIBLE_DEVICES=$GPU \
python $(dirname "$0")/test.py $CONFIG $CHECKPOINT --launcher none $VIS
