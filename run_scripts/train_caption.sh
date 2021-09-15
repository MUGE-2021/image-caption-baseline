TOTAL_NUM_UPDATES=45000
WARMUP_UPDATES=2700
LR=1e-04
BATCH_SIZE=32
UPDATE_FREQ=2
ARCH=muge_baseline_large
DICT_FILE=../utils/cn_tokenizer/dict.txt
VOCAB_FILE=../utils/cn_tokenizer/vocab.txt
DATA_DIR=../dataset/ECommerce-IC
SAVA_DIR=../checkpoints/ECommerce-IC
USER_DIR=../user_module

CUDA_VISIBLE_DEVICES=0,1,2,3 python ../train.py ${DATA_DIR} \
    --dict-file ${DICT_FILE} \
    --vocab-file ${VOCAB_FILE} \
    --save-dir ${SAVA_DIR} \
    --batch-size ${BATCH_SIZE} \
    --task caption \
    --encoder-normalize-before \
    --decoder-normalize-before \
    --layernorm-embedding \
    --patch-layernorm-embedding \
    --share-all-embeddings \
    --share-decoder-input-output-embed \
    --arch ${ARCH} \
    --criterion label_smoothed_cross_entropy \
    --label-smoothing 0.1 \
    --dropout 0.1 --attention-dropout 0.1 \
    --weight-decay 0.01 --optimizer adam --adam-betas "(0.9, 0.999)" --adam-eps 1e-08 \
    --clip-norm 0.1 \
    --lr-scheduler polynomial_decay --lr ${LR} --total-num-update ${TOTAL_NUM_UPDATES} --warmup-updates ${WARMUP_UPDATES} \
    --fp16 --update-freq ${UPDATE_FREQ} \
    --find-unused-parameters \
    --user-dir ${USER_DIR} \
    --log-format 'simple' --log-interval 10 \
    --keep-last-epochs 5 \
    --fixed-validation-seed 7 \
    --save-interval 10 --validate-interval 10 \
    --max-update ${TOTAL_NUM_UPDATES} \
    --num-workers 4

