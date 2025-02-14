



pair=wd_arabic
arch=transformer

lr=0.001
scheduler=warmupinvsqr
max_steps=20000
warmup=4000
beta2=0.98       # 0.999
label_smooth=0.1 # 0.0
total_eval=50
bs=400 # 256

# transformer
layers=4
hs=1024
embed_dim=256
nb_heads=4
dropout=${2:-0.3}

inputPrefix=$1
#modelDir=$2

data_dir=data/NET
ckpt_dir=checkpoints
modelDir=$ckpt_dir/$arch/net-dropout$dropout/$pair

paste <(cut -f1,2 "${inputPrefix}.train80.train64") > "${inputPrefix}.train80.train64.2c"
paste <(cut -f1,2 "${inputPrefix}.train80.dev16") > "${inputPrefix}.train80.dev16.2c"
paste <(cut -f1,2 "${inputPrefix}.test20") > "${inputPrefix}.test20.2c"

python src/train.py \
    --dataset net \
    --train "${data_dir}/${inputPrefix}.train80.train64.2c" \
    --dev "${data_dir}/${inputPrefix}.train80.dev16.2c" \
    --test "${data_dir}/${inputPrefix}.test20.2c" \
    --model $modelDir \
    --embed_dim $embed_dim --src_hs $hs --trg_hs $hs --dropout $dropout --nb_heads $nb_heads \
    --label_smooth $label_smooth --total_eval $total_eval \
    --src_layer $layers --trg_layer $layers --max_norm 1 --lr $lr --shuffle \
    --arch $arch --gpuid 0 --estop 1e-8 --bs $bs --max_steps $max_steps \
    --scheduler $scheduler --warmup_steps $warmup --cleanup_anyway --beta2 $beta2 --bestacc
