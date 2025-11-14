# ----------- training hyperparameters -----------
EPOCHS=100
SEED=42
LEARNING_RATE=1e-5 # 1e-5 5e-6
BATCH_SIZE=16
NO_CH_CUR_EMB=true # true, false
MODEL=rnn # lstm, rnn, transformer
LAYER_NUM=3
TRAIN=TRUE
# ----------- Data hyperparameters -----------
SAMPLE=0 # 20000, 4000, 1000, 0
PREDICT_DATA=true # true, false
CLS_TOKEN=false # true, false
#RESPLIT_DATA=true # true, false
#ONE_TOKEN_PER_DAY=false # true, false
SEQ_LEN=200
#SOFT_LABEL=0
#TRUE_WEIGHT=1  # 1
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
# ======== Stage 1：Data Preprocess ========
echo "========================================"
echo "Step 1: Running dataloader to generate NPZ files..."
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

# ======== Stage 2：training ========
echo "========================================"
echo " Step 2: training model main_train.py ..."
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
    --sample_size $SAMPLE \
    --predict_data $PREDICT_DATA \
    --CLS_token $CLS_TOKEN \
    --seq_len $SEQ_LEN