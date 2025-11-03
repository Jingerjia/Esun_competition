SAMPLE=20000
SEQ_LEN=50
DATA_DIR=datasets/initial_competition/sample_${SAMPLE}_seq_len_${SEQ_LEN}

python main_train.py \
  --train_npz $DATA_DIR/train.npz \
  --val_npz $DATA_DIR/val.npz \
  --test_npz datasets/initial_competition/Esun_test.npz \
  --output_dir checkpoints/transformer \
  --lr 1e-4 \
  --seed 42 \
  --epochs 100 \
  --batch_size 16
