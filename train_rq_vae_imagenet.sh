MASTER_ADDR=127.0.0.1
PORT=29501
SAVE_DIR="./results/rqvae/imagenet256_8x8x4"

torchrun --master_addr=$MASTER_ADDR --master_port=$PORT --nnodes=1 --nproc_per_node=1 --node_rank=0 \
    main_stage1.py -m=configs/imagenet256/stage1/in256-rqvae-8x8x4.yaml -r=$SAVE_DIR



# python -m torch.distributed.launch \
#     --master_addr=$MASTER_ADDR \
#     --master_port=$PORT \
#     --nnodes=1 --nproc_per_node=1 --node_rank=0 \ 
#     main_stage1.py \
#     -m=configs/imagenet256/stage1/in256-rqvae-8x8x4.yaml -r=$SAVE_DIR