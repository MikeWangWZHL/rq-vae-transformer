MASTER_ADDR=127.0.0.1
PORT=29501
SAVE_DIR="./results/rqvae/mnist_4x4x1"

torchrun --master_addr=$MASTER_ADDR --master_port=$PORT --nnodes=1 --nproc_per_node=1 --node_rank=0 \
    main_stage1.py -m=configs/mnist/stage1/in28-rqvae-4x4x1.yaml -r=$SAVE_DIR