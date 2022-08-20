DDIR="../data/opus/wiki-matrix-v1-hi/WikiMatrix/raw/hi"
DDIR_1=$DDIR
PREF="WikiMatrix"
LANG="hi"
# python xml2text.py ${DDIR}/${PREF}.xml ${DDIR}/${PREF}.tsv
# NUM_LINES=$(wc -l "${DDIR}/${PREF}.tsv" | awk '{print $1}')
# python remove-foreign-languages.py ${DDIR}/${PREF}.tsv ${DDIR}/${PREF}.clean.tsv  --nlines $NUM_LINES --lang $LANG
TOKENS="${DDIR}/${PREF}.clean.tokens.tsv"
TOKENS_1=$TOKENS
# python get-tokens.py ${DDIR}/${PREF}.clean.tsv $TOKENS  --nlines $NUM_LINES
DDIR="../data/opus/wiki-matrix-v1-hi/WikiMatrix/raw/en"
DDIR_2=$DDIR
PREF="WikiMatrix"
LANG="en"
# python xml2text.py ${DDIR}/${PREF}.xml ${DDIR}/${PREF}.tsv
# NUM_LINES=$(wc -l "${DDIR}/${PREF}.tsv" | awk '{print $1}')
# python remove-foreign-languages.py ${DDIR}/${PREF}.tsv ${DDIR}/${PREF}.clean.tsv  --nlines $NUM_LINES --lang $LANG
TOKENS="${DDIR}/${PREF}.clean.tokens.tsv"
TOKENS_2=$TOKENS
# python get-tokens.py ${DDIR}/${PREF}.clean.tsv $TOKENS  --nlines $NUM_LINES
DDIR="../data/opus/wikimedia-v20210402-en/wikimedia/raw/en"
DDIR_3=$DDIR
PREF="wikimedia"
LANG="en"
# NUM_LINES=$(wc -l "${DDIR}/${PREF}.xml" | awk '{print $1}')
# python xml2text.py ${DDIR}/${PREF}.xml ${DDIR}/${PREF}.tsv --nlines $NUM_LINES
# NUM_LINES=$(wc -l "${DDIR}/${PREF}.tsv" | awk '{print $1}')
# python remove-foreign-languages.py ${DDIR}/${PREF}.tsv ${DDIR}/${PREF}.clean.tsv  --nlines $NUM_LINES --lang $LANG
TOKENS="${DDIR}/${PREF}.clean.tokens.tsv"
TOKENS_3=$TOKENS
# python get-tokens.py ${DDIR}/${PREF}.clean.tsv $TOKENS  --nlines $NUM_LINES
DDIR="../data/opus/cc-matrix-v1-hi/CCMatrix/raw/hi"
DDIR_4=$DDIR
PREF="CCMatrix"
LANG="hi"
# python xml2text.py ${DDIR}/${PREF}.xml ${DDIR}/${PREF}.tsv
NUM_LINES=$(wc -l "${DDIR}/${PREF}.tsv" | awk '{print $1}')
#python remove-foreign-languages.py ${DDIR}/${PREF}.tsv ${DDIR}/${PREF}.clean.tsv  --nlines $NUM_LINES --lang $LANG
TOKENS="${DDIR}/${PREF}.clean.tokens.tsv"
TOKENS_4=$TOKENS
#python get-tokens.py ${DDIR}/${PREF}.clean.tsv $TOKENS  --nlines $NUM_LINES
DDIR="../data/opus/cc-aligned-hi/CCAligned/raw/hi"
DDIR_5=$DDIR
PREF="CCAligned"
LANG="hi"
# python concat-xml-files.py ${DDIR} ${DDIR}/${PREF}.tsv
# # NUM_LINES=$(wc -l "${DDIR}/${PREF}.tsv" | awk '{print $1}')
# python remove-foreign-languages.py ${DDIR}/${PREF}.tsv ${DDIR}/${PREF}.clean.tsv  --nlines $NUM_LINES --lang $LANG
TOKENS="${DDIR}/${PREF}.clean.tokens.tsv"
TOKENS_5=$TOKENS
#python get-tokens.py ${DDIR}/${PREF}.clean.tsv $TOKENS  --nlines $NUM_LINES


#printf "$TOKENS_2\\n$TOKENS_3" > temp.txt
#python merge-counts.py temp.txt "../data/opus/en-counts.tsv"

#printf "$TOKENS_4\n$TOKENS_5\n$TOKENS_1" > temp.txt
#python merge-counts.py temp.txt "../data/opus/hi-counts.tsv"

#printf "$TOKENS_2\\n$TOKENS_3\\n$TOKENS_4\\n$TOKENS_5\n$TOKENS_1" > temp.txt
#python merge-counts.py temp.txt "../data/opus/all-counts.tsv"

#wc -l ../data/opus/*.tsv
#inputf_en="../data/opus/en-counts"
#inputf_hi="../data/opus/hi-counts"
#NUM_DEV=10000
#python normalize_opus.py "${inputf_en}.tsv" "${inputf_en}.normalized" --lang "en"
#python normalize_opus.py "${inputf_hi}.tsv" "${inputf_hi}.normalized" --lang "hi"
#
#python unify-langs.py "../data/opus/{lang}-counts.normalized" "../data/opus/{lang}-counts.normalized.unify"
#
#
#bash cross_validation_split-90-10.sh "${inputf_en}.normalized.unify"
#mv "${inputf_en}.normalized.unify.test" "${inputf_en}.test10"
#mv "${inputf_en}.normalized.unify.train" "${inputf_en}.train90"
#
#bash cross_validation_split-N.sh "${inputf_en}.train90" $NUM_DEV
##mv "${inputf_en}.train90.train" "${inputf_en}.train90.train"
#mv "${inputf_en}.train90.test" "${inputf_en}.train90.dev${NUM_DEV}"
#
#
#bash cross_validation_split-90-10.sh "${inputf_hi}.normalized.unify"
#mv "${inputf_hi}.normalized.unify.test" "${inputf_hi}.test10"
#mv "${inputf_hi}.normalized.unify.train" "${inputf_hi}.train90"
#
#bash cross_validation_split-N.sh "${inputf_hi}.train90" $NUM_DEV
##mv "${inputf_hi}.train90.train" "${inputf_hi}.train90.train"
#mv "${inputf_hi}.train90.test" "${inputf_hi}.train90.dev${NUM_DEV}"
#
#
#cat "${inputf_hi}.train90.train" "${inputf_en}.train90.train" > "../data/opus/hi-en.train90.train"
#cat "${inputf_hi}.train90.dev${NUM_DEV}" "${inputf_en}.train90.dev${NUM_DEV}" > "../data/opus/hi-en.train90.dev${NUM_DEV}"
#cat "${inputf_hi}.test10" "${inputf_en}.test10" > "../data/opus/hi-en.test10"




