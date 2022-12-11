python examples/gnn/gnn_example_node_classification.py \
    --root_dir ${1}              \
    --graph_name "ogbn-products" \
    --epochs 20                  \
    --batchsize 1024             \
    --classnum 46                \
    --neighbors 5,10,15          \
    --hiddensize 256             \
    --layernum 3                 \
    --framework "wg"             \
    --model "gcn"                \
    --dataloaderworkers 0        \
    --heads 1                    \
    --inferencesample 30         \
    --dropout 0.5                \
    --lr 0.003                   \
    ${2:-""}

# framework: wg, pyg, dgl
# model: gcn, sage, gat
# aggregator: mean, sum