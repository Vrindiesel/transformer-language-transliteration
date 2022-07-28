"""
Created by davan 
7/23/22
"""
import xml.etree.ElementTree as ET
import argparse
import pandas as pd
import os
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input')
    parser.add_argument('output')
    parser.add_argument("--sep", type=str, default="\t")
    parser.add_argument("--lang", type=str, default=None)
    parser.add_argument("--nlines", type=int, default=None)
    args = parser.parse_args()
    print("merging xml file to tsv ...")
    print(args)

    #aggregated = []
    with open(args.output, "w") as fout:
        _fnames = os.listdir(args.input)
        for fname in tqdm(_fnames, total=len(_fnames)):
            if fname.endswith(".xml"):
                fpath = os.path.join(args.input, fname)
                tree = ET.parse(fpath)
                root = tree.getroot()
                #aggregated.extend([{"id": f"{fname}-{child.attrib['id']}", "text": child.text} for child in root])
                for child in root:
                    fout.write(f"{fname}-{child.attrib['id']}{args.sep}{child.text}\n")




    #pd.DataFrame(aggregated).to_csv(args.output, sep="\t", index=False, header=False)


if __name__ == "__main__":
    main()