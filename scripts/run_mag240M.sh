python ../examples/gnn/gnn_examples_rgnn.py \
    --root_dir ${#1} \
    --epochs 80 \
    --batchsize 1024 \
    --classnum 153 \
    --neighbors 25,15 \
    --hiddensize 1024 \
    --layernum 2 \
    --framework wg \ # no pyg here
    --model gcn \ #valid are gcn, sage, gat
    --dataloaderworkers 0 \
    --heads 4 \
    --dropout 0.5 \
    --aggregator "sum" \ # valid mean, sum
    --lr 0.001 \
    --truncate_dim -1 \ # class number
    ${2:-""}
