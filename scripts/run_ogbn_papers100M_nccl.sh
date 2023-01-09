export DATASET_ROOT="/workspace/home/gnn-datasets/ogbn"
NP=${1:-1}

BIND_CMD="/workspace/home/bind.sh --cluster=selene --ib=single --cpu=node"

mpirun --allow-run-as-root --bind-to none -np $NP \
--output-filename /workspace/home/results/wholegraph_nccl_ogbn_papers100M_1_node_${NP}_gpu \
${BIND_CMD} python $WHOLEGRAPH_PATH/examples/gnn/gnn_example_node_classification.py \
    --root_dir $DATASET_ROOT \
    --graph_name ogbn-papers100M \
    --epochs 24 \
    --batchsize 1024 \
    --classnum 172 \
    --neighbors 30,30,30 \
    --hiddensize 256 \
    --layernum 3 \
    --framework "wg" \
    --model "gcn" \
    --dataloaderworkers 0 \
    --heads 4 \
    --inferencesample 30 \
    --dropout 0.5 \
    --lr 0.003 \
    --use_nccl
# framework: wg, pyg, dgl
# model: gcn, sage, gat
# aggregator: mean, sum