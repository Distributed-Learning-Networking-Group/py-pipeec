export TP_SOCKET_IFNAME=enp5s0f1
export GLOO_SOCKET_IFNAME=enp5s0f1

NUM_NODES=2
NUM_TRAINERS=1
JOB_ID=114
HOST_NODE_ADDR=192.168.124.107:11457

# main.py -m vgg16 -s 64 -b 39 naive-128 -a 192.168.124.103:11451

torchrun \
    --nnodes=$NUM_NODES \
    --nproc-per-node=$NUM_TRAINERS \
    --max-restarts=3 \
    --rdzv-id=$JOB_ID \
    --rdzv-backend=c10d \
    --rdzv-endpoint=$HOST_NODE_ADDR \
    main.py -m vgg16 -s 32 naive-128 -a 192.168.124.107:11451 -c 4