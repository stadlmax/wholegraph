export DATASET_ROOT="/workspace/home/gnn-datasets/ogbn"
MODEL=${1:-"sage"}
NP=${2:-1}

BIND_CMD="/workspace/home/bind-luna-multinode.sh --cluster=selene --ib=single --cpu=exclusive"

mpirun --allow-run-as-root --bind-to none -np $NP \
--output-filename /workspace/home/results/wholegraph/fixed_global_bs_ogbn_papers_${MODEL}_1_node_${NP}_gpu \
${BIND_CMD} python $WHOLEGRAPH_PATH/examples/gnn/gnn_example_node_classification.py \
    --root_dir $DATASET_ROOT     \
    --graph_name "ogbn-products" \
    --epochs 24                  \
    --batchsize $((1024 / $NP))  \
    --classnum 46                \
    --neighbors 25,15,5          \
    --hiddensize 256             \
    --layernum 3                 \
    --framework "wg"             \
    --model $MODEL               \
    --dataloaderworkers 0        \
    --heads 4                    \
    --dropout 0.5                \
    --lr 0.003
# framework: wg, pyg, dgl
# model: gcn, sage, gat
