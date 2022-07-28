"""
Created by davan 
7/23/22
"""



import argparse
from tqdm import tqdm
import string
from collections import Counter

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input')
    parser.add_argument('output')
    parser.add_argument("--sep", type=str, default="\t")
    parser.add_argument("--nlines", type=int, default=None)
    args = parser.parse_args()

    print("computing vocabulary from text ...")
    print("args:", args)

    vocab = Counter()
    num_cols = None
    with open(args.input, "r") as fin:
        for line in tqdm(fin, total=args.nlines):
            row = line.split(args.sep)
            if num_cols is None:
                num_cols = len(row)

            if len(row) > 1:
                # take the second column
                text = row[1]
            elif len(row) < 1:
                continue
            else:
                text = row[0]
            vocab.update(Counter(text.strip().split()))

    with open(args.output, "w") as fout:
        for entry in vocab.most_common():
            fout.write(args.sep.join([str(e) for e in entry]))
            fout.write("\n")

    print("vocabulary size:", len(vocab))


if __name__ == "__main__":
    main()





