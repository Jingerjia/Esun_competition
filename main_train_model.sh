# ----------- training hyperparameters -----------
EPOCHS=100
SEED=42
LEARNING_RATE=1e-5 # 1e-5 5e-6
BATCH_SIZE=16
NO_CH_CUR_EMB=true # true, false
MODEL=rnn # lstm, rnn, transformer
LAYER_NUM=3
TRAIN=TRUE
# ----------- Clustering hyperparameters -----------
DO_CLUSTERING=true # true, false
CLUSTER_ANYWAY=true # true, false
CLUSTERING_METHOD=spectral
THRESHOLD=0.5
CLUSTERING_METHOD=gmm # kmeans, gmm
CLUSTERING_SOFT_LABEL=0.2
#CUSTER_NAME="Clustering_${CLUSTERING_METHOD}_${CLUSTERS}_${THRESHOLD}_label_${CLUSTERING_SOFT_LABEL}"
CUSTER_NAME=None
# ----------- Data hyperparameters -----------
SAMPLE=0 # 20000, 4000, 1000, 0
PREDICT_DATA=true # true, false
CLS_TOKEN=false # true, false
RESPLIT_DATA=true # true, false
ONE_TOKEN_PER_DAY=false # true, false
SEQ_LEN=200
SOFT_LABEL=0
TRUE_WEIGHT=1  # 1
TRAIN_RATIO=0.9 # 0.9 0.7

# ----------- Data hyperparameters -----------
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

if [ "${ONE_TOKEN_PER_DAY}" = "true" ]; then
    OTPD="_one_token_per_day"
else
    OTPD=""
fi

if [ "${RESPLIT_DATA}" = "true" ]; then
    TRAIN_NPZ=$DATA_DIR/train${OTPD}_resplit.npz
    VAL_NPZ=$DATA_DIR/val${OTPD}_resplit.npz
else
    TRAIN_NPZ=$DATA_DIR/train${OTPD}.npz
    VAL_NPZ=$DATA_DIR/val${OTPD}.npz
fi

TEST_NPZ=datasets/initial_competition/Esun_test/Esun_test_seq_${SEQ_LEN}${OTPD}.npz
CLUSTERED_TRAIN_NPZ="${DATA_DIR}/train_${CLUSTERING_METHOD}_n${CLUSTERS}_thresh${THRESHOLD}.npz"
# ======== Stage 1：Data Preprocess ========
echo "========================================"
echo "?? Step 1: Running dataloader to generate NPZ files..."
echo "========================================"

python data_preprocess.py \
--sample_size $SAMPLE \
--seq_len $SEQ_LEN \
--seed $SEED \
--one_token_per_day $ONE_TOKEN_PER_DAY \
--predict_data $PREDICT_DATA \
--soft_label $SOFT_LABEL \
--resplit_data $RESPLIT_DATA \
--train_ratio $TRAIN_RATIO

# ======== Stage 2：Clustering ========
echo "========================================"
echo "Step 2:  clustering.py Clustering training data..."
echo "========================================"

if [ "${DO_CLUSTERING}" = "true" ]; then
	if [ ! -f "${CLUSTERED_TRAIN_NPZ}" ] || [ "${CLUSTER_ANYWAY}" = "true" ]; then
	python clustering.py \
        --input_npz ${TRAIN_NPZ} \
        --output_npz ${CLUSTERED_TRAIN_NPZ} \
        --n_clusters ${CLUSTERS} \
        --method ${CLUSTERING_METHOD} \
        --batch_size 128 \
        --threshold $THRESHOLD \
        --soft_label $CLUSTERING_SOFT_LABEL \
        --seed $SEED
	fi

	if [ ! -f "${CLUSTERED_TRAIN_NPZ}" ]; then
	  echo "Clustering fail, output path not found ${CLUSTERED_TRAIN_NPZ}"
	  exit 1
	fi

	echo "Clustering finished, saved in  ${CLUSTERED_TRAIN_NPZ}"
	echo ""
    TRAIN_NPZ=$CLUSTERED_TRAIN_NPZ
fi

if [ ! "${DO_CLUSTERING}" = "true" ]; then
	echo "skip Clustering"
	echo ""
fi

# ======== Stage 3：training ========
echo "========================================"
echo " Step 3: training model main_train.py ..."
echo "========================================"

echo "TRAIN_NPZ: $TRAIN_NPZ"
if [ ! "${TRAIN}" = "TRUE" ]; then
	echo "暫不訓練"
	exit 1
fi

python main_train.py \
    --output_dir checkpoints/$MODEL/$OTPD \
    --train_npz $TRAIN_NPZ \
    --val_npz $VAL_NPZ \
    --test_npz $TEST_NPZ \
    --epochs $EPOCHS \
    --seed $SEED \
    --lr $LEARNING_RATE \
    --batch_size $BATCH_SIZE \
    --model $MODEL \
    --num_layers $LAYER_NUM \
    --without_channel_currency_emb $NO_CH_CUR_EMB \
    --use_cluster $DO_CLUSTERING \
    --cluster_name $CUSTER_NAME \
    --sample_size $SAMPLE \
    --predict_data $PREDICT_DATA \
    --one_token_per_day $ONE_TOKEN_PER_DAY \
    --CLS_token $CLS_TOKEN \
    --seq_len $SEQ_LEN \
    --soft_label $SOFT_LABEL \
    --true_weight $TRUE_WEIGHT \