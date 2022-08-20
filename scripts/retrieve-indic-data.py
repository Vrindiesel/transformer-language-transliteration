"""
Created by davan 
8/2/22
"""

from idatasets import load_devdas
from idatasets import load_occ
from idatasets import load_news_crawl
from idatasets import load_tweets
from idatasets import load_hinglish
from idatasets import load_monolingual
from idatasets import load_wikipedia
from collections import Counter, defaultdict

from spacy.lang.hi import Hindi
nlp = Hindi()

import pyiwn
iwn = pyiwn.IndoWordNet(lang=pyiwn.Language.HINDI)


"""
https://dumps.wikimedia.org/hiwiki/20220320/
https://www.cfilt.iitb.ac.in/iitb_parallel/
https://github.com/anoopkunchukuttan/indic_nlp_library/
https://opus.nlpl.eu/index.php
https://github.com/NirantK/hindi2vec/releases/tag/bbc-hindi-v0.1

still to check:
https://github.com/midas-research/hindi-nli-data
"""
def main():

    loaders = [
        #("indic_devdas", load_devdas),
        #("indic_occ", load_occ),
        #("indic_news_crawl", load_news_crawl),
        #("indic_hinglish", load_hinglish),
        #("indic_monolingual", load_monolingual),
        #("indic_wikipedia", load_wikipedia),
        ("indic_tweets", load_tweets),
    ]

    for name, loader in loaders:
        print(name)
        dataset = loader()
        try:
            paragraphs = list(dataset.data)
        except Exception as e:
            print("error retrieving", name)
            continue

        with open(f"../data/indic/{name}.txt", "w") as fout:
            for text in paragraphs:
                try:
                    doc = nlp(text.decode("utf-8") if isinstance(text, bytes) else text)
                except Exception as e:
                    print("\ntext:", text)
                    raise e
                text = " ".join([token.text for token in doc])
                fout.write(text)
                if "\n" not in set(text):
                    fout.write("\n")



# process text chunks

if __name__ == "__main__":
    main()



