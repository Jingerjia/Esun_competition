
# ----------- 訓練超參數設定 -----------
EPOCHS=100
SEED=42
LEARNING_RATE=1e-5
BATCH_SIZE=16
# ----------- Clustering 參數設定 -----------
GENERATE_ANYWAY=FALSE
CLUSTERS=8
CLUSTER_ANYWAY=FALSE
CLUSTERING_METHOD=kmeans
#DO_CLUSTERING=TRUE
# ----------- Data 參數設定 -----------
SAMPLE=4000
PREDICT_DATA=true
SEQ_LEN=100
SOFT_LABEL=0
LAYER_NUM=3

# ----------- 路徑設定 -----------
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

TRAIN_NPZ=$DATA_DIR/train.npz
CLUSTERED_TRAIN_NPZ=${DATA_DIR}/train_cluster.npz

# ======== 階段一：資料前處理 ========
echo "========================================"
echo "🚀 Step 1: Running dataloader to generate NPZ files..."
echo "========================================"

if [ "$PREDICT_DATA" = true ]; then
  python data_preprocess.py \
  --sample_size $SAMPLE \
  --seq_len $SEQ_LEN \
  --soft_label $SOFT_LABEL \
  --predict_data

else
  python data_preprocess.py \
  --sample_size $SAMPLE \
  --seq_len $SEQ_LEN \
  --soft_label $SOFT_LABEL
fi

# ======== 階段二：Clustering ========
echo "========================================"
echo "🚀 Step 2: 執行 clustering.py 對訓練資料進行聚類 ..."
echo "========================================"

if [ "${DO_CLUSTERING}" = "TRUE" ]; then
	if [ ! -f "${CLUSTERED_TRAIN_NPZ}" ] || [ "${CLUSTER_ANYWAY}" = "TRUE" ]; then
	python clustering.py \
	  --input_npz ${TRAIN_NPZ} \
	  --n_clusters ${CLUSTERS} \
	  --method ${CLUSTERING_METHOD} \
	  --batch_size 128
	fi


	if [ ! -f "${CLUSTERED_TRAIN_NPZ}" ]; then
	  echo "❌ Clustering 失敗，找不到輸出檔案 ${CLUSTERED_TRAIN_NPZ}"
	  exit 1
	fi

	echo "✅ Clustering 完成，已生成 ${CLUSTERED_TRAIN_NPZ}"
	echo ""
fi


if [ ! "${DO_CLUSTERING}" = "TRUE" ]; then
	echo "跳過 Clustering 階段"
	echo ""
fi

# ======== 階段三：模型訓練 ========
echo "========================================"
echo "🚀 Step 3: 開始訓練模型 main_train.py ..."
echo "========================================"

if [ "$PREDICT_DATA" = true ]; then
  python main_train.py \
    --train_npz $TRAIN_NPZ \
    --val_npz $DATA_DIR/val.npz \
    --test_npz datasets/initial_competition/Esun_test/Esun_test_seq_${SEQ_LEN}.npz \
    --output_dir checkpoints/transformer \
    --sample_size $SAMPLE \
    --seq_len $SEQ_LEN \
    --soft_label $SOFT_LABEL \
    --lr $LEARNING_RATE \
    --seed $SEED \
    --epochs $EPOCHS \
    --num_layers $LAYER_NUM \
    --batch_size $BATCH_SIZE \
    --predict_data

else
  python main_train.py \
    --train_npz $TRAIN_NPZ \
    --val_npz $DATA_DIR/val.npz \
    --test_npz datasets/initial_competition/Esun_test/Esun_test_seq_${SEQ_LEN}.npz \
    --output_dir checkpoints/transformer \
    --sample_size $SAMPLE \
    --seq_len $SEQ_LEN \
    --soft_label $SOFT_LABEL \
    --lr $LEARNING_RATE \
    --seed $SEED \
    --epochs $EPOCHS \
    --num_layers $LAYER_NUM \
    --batch_size $BATCH_SIZE
fi