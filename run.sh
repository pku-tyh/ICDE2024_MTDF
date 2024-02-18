DATASETS=(NCI1)
NUM_RUNS=1
LOG_FILE='test.txt'
for SOURCE_INDEX in 0 1 2 3; do
    for TARGET_INDEX in 0 1 2 3; do
        if [ $SOURCE_INDEX != $TARGET_INDEX ]; then
            python main.py --DS $DATASETS --log_file $LOG_FILE \
                --epochs 100 --number_of_run $NUM_RUNS --use_bn 1 --batch_size 1024 --lr 0.001 --sdfa_lr 0.001\
                --hidden-dim 128 --cross_dataset 0 --conv_type GCN\
                --data_split 4 --source_index $SOURCE_INDEX --target_index $TARGET_INDEX \
                --eval_interval 5 --target_epochs 30 --sdfa_eval_interval 1 \
                --mixup_weight_target 0.3 --mixup_weight_source 0.9 
        fi
    done
done
