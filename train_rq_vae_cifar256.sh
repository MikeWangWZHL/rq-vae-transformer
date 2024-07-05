MASTER_ADDR=127.0.0.1
PORT=29502
SAVE_DIR="./results/rqvae/cifar100_128_rqvae_single_level_mse_only"
CONFIG_FILE="configs/cifar100/cifar100_256_rqvae_single_level_mse_only.yaml"

RUN_NAME="sanity_check_rqvae_single-level_cifar100_256_tok-64_cb-16384_mse-only"
# export SMOKE_TEST=1
torchrun --master_addr=$MASTER_ADDR --master_port=$PORT --nnodes=1 --nproc_per_node=1 --node_rank=0 \
    main_stage1.py -m=$CONFIG_FILE -r=$SAVE_DIR --run_name=$RUN_NAME