"""
Created by davan 
7/25/22
"""

import xml.etree.ElementTree as ET
import argparse
import pandas as pd
import os
from tqdm import tqdm
from collections import Counter, defaultdict

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input')
    parser.add_argument('output')
    parser.add_argument("--sep", type=str, default="\t")
    parser.add_argument("--lang", type=str, default=None)
    parser.add_argument("--mf_hi", type=int, default=4)
    parser.add_argument("--mf_en", type=int, default=20)
    args = parser.parse_args()
    print("building dataset ...")
    print(args)

    all_df = defaultdict(dict)
    all_counts = Counter()
    min_freq = {"hi": args.mf_hi, "en": args.mf_en}
    the_langs = ["hi", "en"]


    df = pd.read_csv(args.input, sep=args.sep, header=None, names=["word", "freq"])
    df["w_len"] = df.word.apply(lambda x: len(str(x)))

    print("Raw Dataset Size:", len(df))
    for thresh in [1, 2, 3, 4, 5, 6, 7, 8, 10, 20]:
        subset = df.loc[df.freq>=thresh]
        subset_len = len(subset)
        print(f" * subset(N={thresh}) size: {subset_len} %{100*subset_len/len(df)}")
        for size in [2, 3, 4, 5, 6]:
            ss = subset.loc[subset.w_len>=size]
            print(f"     * subset(length={size}) size: {len(ss)} %{100 * len(ss) / subset_len}")


    df = df.loc[(df.freq>=min_freq[args.lang])&(df.w_len>=4)]
    lang_counts = Counter({word: freq for word,freq in zip (df.word, df.freq)})

    with open(args.output, "w") as fout:
        for word, freq in lang_counts.most_common():
            # skip words that occur in both languages ~ davan
            #if word not in all_df[f"{other_lang[args.lang]}_counts"]:
            fout.write(f"{word}{args.sep}{freq}\n")




if __name__ == "__main__":
    main()
