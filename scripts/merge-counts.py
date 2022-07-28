"""
Created by davan 
7/25/22
"""

import xml.etree.ElementTree as ET
import argparse
import pandas as pd
import os
from tqdm import tqdm
from collections import Counter

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input')
    parser.add_argument('output')
    parser.add_argument("--sep", type=str, default="\t")
    args = parser.parse_args()
    print("merging vocabulary files ...")
    print(args)

    with open(args.input, "r") as fin:
        fpaths = fin.read().split("\n")

    #aggregated = []
    all_counts = Counter()
    for fpath in fpaths:
        print(f" * merging {fpath}")
        with open(fpath, "r") as fin:
            d = {}
            for entry in fin.read().split("\n"):
                if not entry:
                    continue
                kv = entry.split(args.sep)
                if len(kv) == 2:
                    d[kv[0]] = int(kv[1])
                else:
                    print(f"skipping {kv}")
            #all_counts.update(dict([entry.split(args.sep) for entry in fin.read().split("\n")]))
            all_counts.update(d)

    with open(args.output, "w") as fout:
        for term, freq in all_counts.most_common():
            fout.write(f"{term}{args.sep}{freq}\n")





    #pd.DataFrame(aggregated).to_csv(args.output, sep="\t", index=False, header=False)


if __name__ == "__main__":
    main()


