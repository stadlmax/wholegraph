export DATASET_ROOT="/workspace/home/gnn-datasets/ogbn"
NP=${1:-1}

BIND_CMD="/workspace/home/bind.sh --cluster=selene --ib=single --cpu=node"

mpirun --allow-run-as-root --bind-to none -np $NP \
--output-filename /workspace/home/results/wholegraph_nccl_ogbn_mag240M_1_node_${NP}_gpu \
${BIND_CMD} python $WHOLEGRAPH_PATH/examples/gnn/gnn_examples_rgnn.py \
    --root_dir $DATASET_ROOT \
    --epochs 80 \
    --batchsize 1024 \
    --classnum 153 \
    --neighbors 25,15 \
    --hiddensize 1024 \
    --layernum 2 \
    --framework "wg" \
    --model "gcn" \
    --dataloaderworkers 0 \
    --heads 4 \
    --dropout 0.5 \
    --aggregator "sum" \
    --lr 0.001 \
    --truncate_dim -1 \
    --use_nccl
# framework: wg, dgl
# model: gcn, sage, gat
# aggregator: mean, sum