"""
Created by davan 
7/28/22
"""





import argparse
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input')
    parser.add_argument('output')
    parser.add_argument("--sep", type=str, default="\t")
    parser.add_argument("--nlines", type=int, default=None)
    args = parser.parse_args()

    print("args:", args)

    with open(args.input, "r") as fin, open(args.output, "w") as fout:
        for line in tqdm(fin, total=args.nlines):
            row = line.split(args.sep)
            fout.write(f"{row[0]}{args.sep}{' '.join(row[1].strip().split())}")





if __name__ == "__main__":
    main()

