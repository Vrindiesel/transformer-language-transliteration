
import xml.etree.ElementTree as ET
import argparse
import pandas as pd
import os
from tqdm import tqdm
from collections import Counter, defaultdict




def word_to_sequence(word, lang):
    LANG_TOKEN = {"en": "<en>", "hi": "<hi>"}
    # convert string to list of chars
    return [LANG_TOKEN[lang]] + list(word)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input')
    parser.add_argument('output')
    parser.add_argument("--sep", type=str, default="\t")
    parser.add_argument("--lang", type=str, default=None)
    args = parser.parse_args()
    print("building dataset ...")
    print(args)

    en_df = pd.read_csv(args.input.format(lang="en"), sep=args.sep, header=None, names=["word", "freq"])
    hi_df = pd.read_csv(args.input.format(lang="hi"), sep=args.sep, header=None, names=["word", "freq"])
    hi_words = set(hi_df.word)
    en_counts = Counter({word: freq for word,freq in zip (en_df.word, en_df.freq) if word not in hi_words})
    hi_counts = Counter({word: freq for word,freq in zip (hi_df.word, hi_df.freq) })
    for lang, counts in [("en", en_counts), ("hi", hi_counts)]:
        with open(args.output.format(lang=lang), "w") as fout:
            for word, freq in counts.most_common():
                # skip words that occur in both languages ~ davan
                #if word not in all_df[f"{other_lang[args.lang]}_counts"]:
                seq = word_to_sequence(word, lang)
                fout.write(" ".join(seq))
                fout.write("\n")




if __name__ == "__main__":
    main()
