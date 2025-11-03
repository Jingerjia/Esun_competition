SAMPLE=20000
SEQ_LEN=200
DATA_DIR=datasets/initial_competition/sample_${SAMPLE}_seq_len_${SEQ_LEN}

python main_train.py \
  --Sample $SAMPLE \
  --Sequence $SEQ_LEN \
  --train_npz $DATA_DIR/train.npz \
  --val_npz $DATA_DIR/val.npz \
  --test_npz datasets/initial_competition/Esun_test.npz \
  --num_layers 6 \
  --output_dir checkpoints/transformer \
  --lr 1e-5 \
  --seed 42 \
  --epochs 100 \
  --batch_size 16
