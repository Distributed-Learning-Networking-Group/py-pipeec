NUM_NODES=1
NUM_TRAINERS=2
JOB_ID=114
HOST_NODE_ADDR=localhost:11457

torchrun \
    --nnodes=$NUM_NODES \
    --nproc-per-node=$NUM_TRAINERS \
    --max-restarts=3 \
    --rdzv-id=$JOB_ID \
    --rdzv-backend=c10d \
    --rdzv-endpoint=$HOST_NODE_ADDR \
    main.py -m vgg16 -s 8 -b 18,21 naive-128