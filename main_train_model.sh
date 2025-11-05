SAMPLE=4000
#PREDICT_DATA=true
SEQ_LEN=100
SOFT_LABEL=0

if [ "$PREDICT_DATA" = true ]; then
    SAMPLE_TYPE="predict_data"
else
    SAMPLE_TYPE="sample_${SAMPLE}"
fi

if (( $(echo "$SOFT_LABEL > 0" | bc -l) )); then
    DATA_DIR="datasets/initial_competition/${SAMPLE_TYPE}/${SAMPLE_TYPE}_seq_len_${SEQ_LEN}_soft_label_${SOFT_LABEL}"
else
    DATA_DIR="datasets/initial_competition/${SAMPLE_TYPE}/${SAMPLE_TYPE}_seq_len_${SEQ_LEN}"
fi

echo "DATA_DIR=$DATA_DIR"


if [ "$PREDICT_DATA" = true ]; then

  python data_preprocess.py \
  --sample_size $SAMPLE \
  --seq_len $SEQ_LEN \
  --soft_label $SOFT_LABEL \
  --predict_data

  python main_train.py \
    --train_npz $DATA_DIR/train.npz \
    --val_npz $DATA_DIR/val.npz \
    --test_npz datasets/initial_competition/Esun_test/Esun_test_seq_${SEQ_LEN}.npz \
    --output_dir checkpoints/transformer \
    --sample_size $SAMPLE \
    --seq_len $SEQ_LEN \
    --soft_label $SOFT_LABEL \
    --lr 1e-5 \
    --seed 42 \
    --epochs 100 \
    --num_layers 3 \
    --batch_size 16 \
    --predict_data
else

  python data_preprocess.py \
  --sample_size $SAMPLE \
  --seq_len $SEQ_LEN \
  --soft_label $SOFT_LABEL

  python main_train.py \
    --train_npz $DATA_DIR/train.npz \
    --val_npz $DATA_DIR/val.npz \
    --test_npz datasets/initial_competition/Esun_test/Esun_test_seq_${SEQ_LEN}.npz \
    --output_dir checkpoints/transformer \
    --sample_size $SAMPLE \
    --seq_len $SEQ_LEN \
    --soft_label $SOFT_LABEL \
    --lr 1e-5 \
    --seed 42 \
    --epochs 100 \
    --num_layers 3 \
    --batch_size 16
fi