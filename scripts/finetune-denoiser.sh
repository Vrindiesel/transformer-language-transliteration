


#arch=dntransformer
#arch=fdntransformer
arch=transformer

lr=0.0001
scheduler=warmupinvsqr
max_steps=32_000
warmup=4_000
beta2=0.98       # 0.999
label_smooth=0.1 # 0.0
total_eval=200
bs=500 # 256
SAVE_STEPS=63

# transformer
layers=6
hs=1024
embed_dim=352
nb_heads=8
dropout=${2:-0.3}
MASK_PROB=0.15
MMASK_PROB=0.8
MRAND_PROB=0.15

PREFIX="m1.2_hi-en"
#modelDir=$2
base_name=en-hi-CDM-100K

data_dir="../data/ITD/en_hi"
ckpt_dir="../checkpoints"
#modelDir="$ckpt_dir/mega_en_hi/$arch/${PREFIX}-dr$dropout-l$layers-h${hs}-e${embed_dim}-a${nb_heads}-NL-mask-${MASK_PROB}-${MMASK_PROB}-${MRAND_PROB}-ft-dakshina"
modelDir="$ckpt_dir/mega_en_hi/$arch/${PREFIX}-dr$dropout-l$layers-h${hs}-e${embed_dim}-a${nb_heads}-lm-${MASK_PROB}-${MMASK_PROB}-${MRAND_PROB}-ft-dakshina"



python ../src/dn_train.py \
    --dataset dakshina \
    --train "${data_dir}/${base_name}.train80.train64" \
    --dev "${data_dir}/${base_name}.train80.dev16" \
    --test "${data_dir}/${base_name}.test20" \
    --model $modelDir \
    --arch $arch \
    --embed_dim $embed_dim --src_hs $hs --trg_hs $hs --dropout $dropout --nb_heads $nb_heads \
    --total_eval $total_eval \
    --src_layer $layers --trg_layer $layers --max_norm 1 --lr $lr --shuffle \
    --gpuid 0 --estop 1e-8 --bs $bs --max_steps $max_steps \
    --scheduler $scheduler --warmup_steps $warmup --beta2 $beta2  \
    --share_embed --optimizer adam  --seed 2342 \
    --bestacc \
    --max_decode_len 55 --max_seq_len 50 \
    --label_smooth $label_smooth \
    --init ../checkpoints/mega_en_hi/dntransformer/hi-en-dr0.3-l6-h1024-e352-a8-lm-0.15-0.8-0.15.nll_2.2523.acc_0.6179347826086956.P_0.R_0.P_0.F_0.wacc_34.645.step_87000 \
    --vocab_file "../checkpoints/mega_en_hi/dntransformer/hi-en-dr0.3-l6-h1024-e352-a8-lm-0.15-0.8-0.15.vocab.json" \
    --cleanup_anyway \
    --finetuning \
    --eval_steps ${SAVE_STEPS} \






    #--vocab_file "../checkpoints/transformer/hi-en-dr0.3-l5-h1024-e354-a6-Nomask-0.2-0.75-0.15.vocab.json" \
#     --load_eval \
#     --load ../checkpoints/fdntransformer/hi-en-dr0.3-l5-h1024-e354-a6-NL-mask-0.2-0.75-0.15-ft-dakshina.nll_1.0746.acc_45.8556.meanfs_0.9348.step_19434 \
# checkpoints/mega_en_hi/dntransformer/hi-en-dr0.3-l6-h1024-e352-a8-lm-0.15-0.8-0.15.nll_2.2523.acc_0.6179347826086956.P_0.R_0.P_0.F_0.wacc_34.645.step_87000

    #--finetuning \
    #--vocab_file "../checkpoints/transformer/hi-en-dr0.3-l5-h1024-e354-a6-Nomask-0.2-0.75-0.15.vocab.json" \
    #--init "../checkpoints/transformer/hi-en-dr0.3-l5-h1024-e354-a6-Nomask-0.2-0.75-0.15.nll_1.0895.acc_37.1193.meanfs_0.9503.step_99100" \
    #--load ../checkpoints/fdntransformer/hi-en-dr0.3-l4-h1024-e252-a12-NL-mask-0.3-0.6-0.3-ft-dakshina.nll_0.2864.DevAcc_91.3465step_16906
    #--load "../checkpoints/dntransformer/hi-en-dr0.3-l6-h512-e264-a8-NL-mask-0.3-0.4-0.4.nll_1.3683.DevAcc_80.2421step_23000"






