

# ----------- training hyperparameters -----------
EPOCHS=100
SEED=42
LEARNING_RATE=1e-5 # 1e-5 5e-6
BATCH_SIZE=16
NO_CH_CUR_EMB=true # true, false
MODEL=rnn # rnn
DATA_GEN=TRUE   # TRUE, FASLE
CACHE_GEN=TRUE # TRUE, FASLE
TRAIN=TRUE     # TRUE, FASLE
REPRODUCE=TRUE #是否重現提交之實驗結果
# ----------- Data hyperparameters -----------
SAMPLE=0 # 20000, 4000, 1000, 0
PREDICT_DATA=true # true, false
SEQ_LEN=200
TRAIN_RATIO=0.9 # 0.9 0.7


# ----------- Path hyperparameters -----------
if [ "$PREDICT_DATA" = true ]; then
    SAMPLE_TYPE="predict_data"
else
    SAMPLE_TYPE="sample_${SAMPLE}"
fi

DATA_DIR=datasets/initial_competition/${SAMPLE_TYPE}/${SAMPLE_TYPE}_seq_len_${SEQ_LEN}/train_ratio_$TRAIN_RATIO
TEST_DIR=datasets/initial_competition/Esun_test

echo "DATA_DIR=$DATA_DIR"

if [ "${REPRODUCE}" = "TRUE" ]; then
    echo "重現實驗結果"
    TRAIN_NPZ=results/rnn/predict_data/train.npz
    VAL_NPZ=results/rnn/predict_data/val.npz
else
    TRAIN_NPZ=$DATA_DIR/train.npz
    VAL_NPZ=$DATA_DIR/val.npz
fi

TEST_NPZ=$TEST_DIR/Esun_test_seq_${SEQ_LEN}.npz
# ======== Stage 1：Data Preprocess ========
echo "========================================"
echo "Step 1: Running dataloader to generate NPZ files..."
echo "========================================"

if [ "${CACHE_GEN}" = "TRUE" ]; then
    echo "快取生成中，若已生成請於 main.sh 將 CACHE_GEN 改成 FALSE"
    sleep 5
	python -m Preprocess.preprocess_cache
	python -m Preprocess.acct_data	
else
    echo "跳過快取生成"
fi

if [ "${DATA_GEN}" = "TRUE" ]; then
    python -m Preprocess.data_preprocess \
    --data_dir $DATA_DIR \
    --test_dir $TEST_DIR \
    --train_ratio $TRAIN_RATIO \
    --predict_data $PREDICT_DATA \
    --sample_size $SAMPLE \
    --seq_len $SEQ_LEN \
    --seed $SEED
else
    echo "跳過資料生成"
fi

# ======== Stage 2：training ========
echo "========================================"
echo " Step 2: training model main_train.py ..."
echo "========================================"

echo "TRAIN_NPZ: $TRAIN_NPZ"
if [ ! "${TRAIN}" = "TRUE" ]; then
	echo "暫不訓練"
	exit 1
fi

python -m Model.train \
    --output_dir checkpoints/$MODEL/$OTPD \
    --train_npz $TRAIN_NPZ \
    --val_npz $VAL_NPZ \
    --test_npz $TEST_NPZ \
    --epochs $EPOCHS \
    --seed $SEED \
    --lr $LEARNING_RATE \
    --batch_size $BATCH_SIZE \
    --model $MODEL \
    --sample_size $SAMPLE \
    --predict_data $PREDICT_DATA \
    --seq_len $SEQ_LEN