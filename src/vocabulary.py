"""
Created by davan 
7/28/22
"""
import json

class Vocabulary(object):
    def __init__(self, vocab=None, size=None, c2i=None, from_file=None):
        self.vocab = {} if vocab is None else vocab
        self.size = {} if size is None else size
        self.c2i = {} if c2i is None else c2i

        if from_file:
            self.load(from_file)

    def build(self):
        for name, entries in self.vocab.items():
            self.c2i[name] = {c: i for i, c in enumerate(entries)}
            self.size[name] = len(entries)

    def add_vocab(self, name, ordered_entries):
        self.vocab[name] = ordered_entries
        self.size[name] = len(ordered_entries)

    def save(self, outfile="vocabulary.json"):
        d = {
            "vocab": self.vocab,
            "size": self.size
        }
        with open(outfile, "w") as fout:
            json.dump(d, fout, indent=2)

    def load(self, infile):
        with open(infile, "r") as fin:
            data = json.load(fin)
        self._from_dict(data)

    def _from_dict(self, d):
        for name, entries in d["vocab"].items():
            self.vocab[name] = entries
        self.build()

    @classmethod
    def from_dict(cls, d):
        v = cls()
        v._from_dict(d)
        return v






