"""
Created by davan 
7/23/22
"""


import xml.etree.ElementTree as ET
import argparse
import pandas as pd
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input')
    parser.add_argument('output')
    parser.add_argument("--sep", type=str, default="\t")
    parser.add_argument("--lang", type=str, default="hi")
    parser.add_argument("--nlines", type=int, default=None)
    args = parser.parse_args()
    print("converting xml file to tsv ...")
    print(args)
    tree = ET.parse(args.input)
    root = tree.getroot()
    with open(args.output, "w") as fout:
        for child in tqdm(root, total=len(root)):
            fout.write(f"{child.attrib['id']}{args.sep}{child.text}\n")

    #df = pd.DataFrame([{"id": child.attrib["id"], "text": child.text} for child in root])
    #df.to_csv(args.output, sep=args.sep, index=False, header=False)


if __name__ == "__main__":
    main()

