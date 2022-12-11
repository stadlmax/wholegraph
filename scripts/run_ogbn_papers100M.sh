python ./examples/gnn/gnn_examples_node_classification.py \
    --root_dir ${#1} \
    --graph_name ogbn-papers100M \
    --epochs 24 \
    --batchsize 1024 \
    --classnum 172 \
    --neighbors 30,30,30 \
    --hiddensize 256 \
    --layernum 3 \
    --framework wg \ #wg, pyg, dgl
    --model gcn \ # valid are gcn, sage, gat
    --dataloaderworkers 0 \
    --heads 1 \
    --inferencesample 30 \
    --dropout 0.5 \
    --aggregator "sum" \ # valid mean, sum
    --lr 0.003 \
    ${2:-""}
