#!/bin/bash

# ----------- 參數設定 -----------
SAMPLE=4000
SEQ_LEN=50
GENERATE_ANYWAY=FALSE
EPOCHS=100
CLUSTERS=8
LABEL_THRESHOLD=0.3
CLUSTER_ANYWAY=FALSE
CLUSTERING_METHOD=kmeans
SEED=42
DO_CLUSTERING=TRUE
# ----------- 路徑設定 -----------
DATA_DIR=datasets/initial_competition/sample_${SAMPLE}_seq_len_${SEQ_LEN}
TRAIN_NPZ=${DATA_DIR}/train.npz
VAL_NPZ=${DATA_DIR}/val.npz
OUTPUT_DIR=checkpoints/transformer
TEST_NPZ=datasets/initial_competition/Esun_test.npz
CLUSTERED_TRAIN_NPZ=${DATA_DIR}/train_cluster.npz


# ======== 階段一：資料前處理 ========
echo "========================================"
echo "🚀 Step 1: Running dataloader to generate NPZ files..."
echo "========================================"

if [ ! -d "${DATA_DIR}" ] || [ "${GENERATE_ANYWAY}" = "TRUE" ]; then
python data_preprocess.py \
  --sample_size ${SAMPLE} \
  --seq_len ${SEQ_LEN} \
  --data_dir ${DATA_DIR} \
  --seed ${SEED} \
  --train_val_gen
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
	  --batch_size 128 \
	  --threshold ${LABEL_THRESHOLD}
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

python main_train.py \
  --Sample ${SAMPLE} \
  --Sequence ${SEQ_LEN} \
  --train_npz ${TRAIN_NPZ} \
  --val_npz ${VAL_NPZ} \
  --test_npz ${TEST_NPZ} \
  --num_layers 3 \
  --output_dir ${OUTPUT_DIR} \
  --lr 1e-5 \
  --seed ${SEED} \
  --epochs ${EPOCHS} \
  --batch_size 16 \
  --use_cluster

