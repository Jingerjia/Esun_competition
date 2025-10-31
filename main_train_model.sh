python main_train.py \
  --train_json analyze_UI/cache/train.npz \
  --val_json analyze_UI/cache/val.npz \
  --test_json analyze_UI/cache/val.npz \
  --output_dir checkpoints/transformer \
  --epochs 100 --batch_size 16 --lr 1e-4
