for MODEL in "gat" "sage"
do
    for NP in 1 2 4 8
    do 
        $1 $MODEL $NP
    done
done
