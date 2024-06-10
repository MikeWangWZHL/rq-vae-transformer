NUM_GPU=$1
MASTER_ADDR=127.0.0.1
PORT=29502
SAVE_DIR="./results/rqvae/imagenet256_8x8x4"
CONFIG_FILE="configs/imagenet256_repro/stage1/in256-rqvae-8x8x4.yaml"

echo "Using NUM_GPU: $NUM_GPU"

# export SMOKE_TEST=1
torchrun --master_addr=$MASTER_ADDR --master_port=$PORT --nnodes=1 --nproc_per_node=$NUM_GPU --node_rank=0 \
    main_stage1.py -m=$CONFIG_FILE -r=$SAVE_DIR



# SAVE_DIR="./results/rqvae/imagenet256_8x8x4"
# python -m torch.distributed.launch \
#     --master_addr=$MASTER_ADDR \
#     --master_port=$PORT \
#     --nnodes=1 --nproc_per_node=1 --node_rank=0 \ 
#     main_stage1.py \
#     -m=configs/imagenet256/stage1/in256-rqvae-8x8x4.yaml -r=$SAVE_DIR