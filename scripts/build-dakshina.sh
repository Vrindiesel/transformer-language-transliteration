
DPATH="../data/HI/dakshina_dataset_v1.0/hi/lexicons/"
base_name=hi.translit.sampled

## declare an array variable
declare -a arr=("train" "dev" "test")

## now loop through the above array
for part in "${arr[@]}"
do
  inputFile="${DPATH}/${base_name}.${part}.tsv"
  bash swap2cols.sh ${inputFile}
  python normalize_text.py "${inputFile}2"  "${inputFile}.normalized"
  python align_tokens.py "${inputFile}.normalized" "${inputFile}.normalized.aligned" 1
  python split_pronunciation.py "${inputFile}.normalized.aligned" "${inputFile}.normalized.aligned.tokens"
done

# You can access them using echo "${arr[0]}", "${arr[1]}" also


