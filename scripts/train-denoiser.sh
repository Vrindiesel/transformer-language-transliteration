
#pair=wd_arabic
arch=dntransformer
#arch=transformer

lr=0.0001
scheduler=warmupinvsqr
max_steps=250_000
warmup=4_000
beta2=0.98       # 0.999
label_smooth=0.1 # 0.0
total_eval=1000
bs=400 # 256
SAVE_STEPS=1000

# transformer
layers=6
hs=1024
embed_dim=352
nb_heads=8
dropout=${2:-0.3}
MASK_PROB=0.15
MMASK_PROB=0.8
MRAND_PROB=0.15

PREFIX="hi-en"
#modelDir=$2

data_dir="../data/mega_en_hi"
ckpt_dir="../checkpoints"
modelDir="$ckpt_dir/mega_en_hi/$arch/${PREFIX}-dr$dropout-l$layers-h${hs}-e${embed_dim}-a${nb_heads}-lm-${MASK_PROB}-${MMASK_PROB}-${MRAND_PROB}"

#modelDir="./temp-model"


python ../src/dn_train.py \
    --dataset denoise \
    --train "${data_dir}/${PREFIX}.train90.train" \
    --dev "${data_dir}/${PREFIX}.train90.dev10000" \
    --test "${data_dir}/${PREFIX}.test10" \
    --model $modelDir \
    --embed_dim $embed_dim --src_hs $hs --trg_hs $hs --dropout $dropout --nb_heads $nb_heads \
    --total_eval $total_eval \
    --src_layer $layers --trg_layer $layers --max_norm 1 --lr $lr --shuffle \
    --arch $arch --gpuid 0 --estop 1e-8 --bs $bs --max_steps $max_steps \
    --scheduler $scheduler --warmup_steps $warmup --beta2 $beta2  \
    --mask_prob $MASK_PROB  --mask_mask_prob $MMASK_PROB --mask_random_prob $MRAND_PROB   \
    --share_embed --optimizer adam --max_decode_len 50 --max_seq_len 50 --seed 2342 \
    --eval_steps ${SAVE_STEPS} \
    --bestacc \
    --cleanup_anyway \
    --label_smooth $label_smooth \
    --mask_loss


    #--load "../checkpoints/dntransformer/hi-en-dr0.3-l6-h512-e264-a8-NL-mask-0.3-0.4-0.4.nll_1.3683.DevAcc_80.2421step_23000"

