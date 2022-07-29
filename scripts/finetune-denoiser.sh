


#arch=dntransformer
arch=fdntransformer

lr=0.001
scheduler=warmupinvsqr
max_steps=20_000
warmup=4_000
beta2=0.98       # 0.999
label_smooth=0.1 # 0.0
total_eval=50
bs=400 # 256
SAVE_STEPS=300

# transformer
layers=4
hs=1024
embed_dim=252
nb_heads=12
dropout=${2:-0.3}
MASK_PROB=0.3
MMASK_PROB=0.6
MRAND_PROB=0.3

PREFIX="hi-en"
#modelDir=$2
base_name=en-hi-CDM-100K

data_dir="../data/ITD/en_hi"
ckpt_dir="../checkpoints"
modelDir="$ckpt_dir/$arch/${PREFIX}-dr$dropout-l$layers-h${hs}-e${embed_dim}-a${nb_heads}-NL-mask-${MASK_PROB}-${MMASK_PROB}-${MRAND_PROB}-ft-dakshina"




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
    --share_embed --optimizer adam --max_decode_len 55 --max_seq_len 50 --seed 2342 \
    --bestacc \
    --label_smooth $label_smooth \
    --cleanup_anyway \
    --init "../checkpoints/dntransformer/hi-en-dr0.3-l4-h1024-e256-a8-NL-mask-0.3-0.6-0.3.nll_0.8801.DevAcc_72.7131step_31000" \
    --finetuning \
    --vocab_file "temp-model.vocab.json" \
    --load ../checkpoints/fdntransformer/hi-en-dr0.3-l4-h1024-e252-a12-NL-mask-0.3-0.6-0.3-ft-dakshina.nll_0.2864.DevAcc_91.3465step_16906

    #--eval_steps ${SAVE_STEPS} \



    #--load "../checkpoints/dntransformer/hi-en-dr0.3-l6-h512-e264-a8-NL-mask-0.3-0.4-0.4.nll_1.3683.DevAcc_80.2421step_23000"






