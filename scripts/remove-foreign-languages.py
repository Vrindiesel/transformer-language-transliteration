"""
Created by davan 
7/23/22
"""

from inltk.inltk import remove_foreign_languages
from inltk.inltk import tokenize as inltk_tokenize
import argparse
from tqdm import tqdm
import string
import nltk
from nltk.tokenize import WordPunctTokenizer
import re


try:
    remove_foreign_languages("test string", "hi")
except Exception as e:
    from inltk.inltk import setup
    print("setting up inltk for first time use ...")
    setup("hi")

UNK = "<unk>"
EN_TOKENIZER = WordPunctTokenizer()


PUNCT_LIST = { ',', '.', '!', ';', '_', ":", "{", "}", "[", "]", "-", "–", "|", "·", "・", "‧", "・", "&",
               '“', '…', '“'}
pp = set("’‘′“…“▪】‬‪प�”‘‘‎—​ॱ﻿")

PUNCT_LIST = set(string.punctuation) | PUNCT_LIST | pp
FLIP_PUNC = ','
def fix_flip(input):
    """
    Flips the first-last ordering of a name.
    "HARRISON, DAVAN" ==> "DAVAN HARRISON"

    """
    idx = input.find(FLIP_PUNC)
    if (idx < 0):
        return input
    else:
        return input[idx+1:].strip() + " " + input[:idx]


def isScript(s, script="latin"):
    try:
        s.encode(encoding='utf-8').decode(script)
    except UnicodeDecodeError:
        return False
    else:
        return True

"""
>>> int(hex(ord(c)), 16)
2431
>>> chr(int(hex(ord(c)), 16))
'ॿ'
"""

def is_dev(c):
    """ Devanagari 0x900 - 0x97f  """
    #print("\n", c)
    return c and int("0x900", 16) <= ord(c) <= int("0x97f", 16)

def is_latin(t):
    return t and t.isascii() and t.isalpha()

def clean_these_toks(text, lang="hi", tokenizer=None, strict=False):
    scrubber = is_dev if lang == "hi" else is_latin
    logic_func = all if strict else any
    if tokenizer is not None:
        tokens = inltk_tokenize(text, lang)
        clean_toks = [tok for tok in tokens if logic_func(scrubber(c) for c in tok)]
    else:
        #tokens = EN_TOKENIZER.tokenize(fix_flip(text))
        # split on non-word characters

        #clean_text = text.translate({ord("_"): " ", ord("-"): "_"}).lower()
        #print("tokens:", tokens)
        if lang == "hi":
            tokens = text.translate({ord(x): ' ' for x in PUNCT_LIST}).lower().split()
            clean_toks = [tok for tok in tokens if logic_func(scrubber(c) for c in tok)]
        else:
            tokens = re.split(r'\W', text.lower())
            clean_toks = [tok for tok in tokens if scrubber(tok)]
        #tokens = line.translate({ord(x): ' ' for x in PUNCT_LIST}).lower().split()

    #print("\ntext:", text)
    #print("tokens:", tokens)
    #print("clean_toks:", clean_toks)
    #input(">>>")

    #clean_toks = []
    #unwanted = {UNK, "▁", ""}
    #for tok in tokens:
    #    if tok in unwanted:
    #        continue
    #    if tok.startswith("▁"):
    #        tok = tok[1:]
    #    tok = tok.strip()
    #    if not isScript(tok, "latin"):
    #        clean_toks.append(tok)
    return " ".join(clean_toks)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input')
    parser.add_argument('output')
    parser.add_argument("--sep", type=str, default="\t")
    parser.add_argument("--lang", type=str, default="hi")
    parser.add_argument("--nlines", type=int, default=None)
    args = parser.parse_args()

    print("removing foreign language tokens from text ...")
    print("args:", args)

    with open(args.input, "r") as fin, open(args.output, "w") as fout:
        for line in tqdm(fin, total=args.nlines):
            row = line.split(args.sep)
            clean = [row[0]]
            for col_val in row[1:]:
                #print("dirty_vals:")
                #print(col_val)
                #clean_vals = remove_foreign_languages(col_val, args.lang)
                #print("clean_vals:")
                #print(clean_vals)

                clean_vals = clean_these_toks(col_val, strict=True)
                #print("clean_vals:")
                #print(clean_vals)

                clean.append(clean_vals)

            fout.write(args.sep.join(clean))
            fout.write("\n")


if __name__ == "__main__":
    main()
