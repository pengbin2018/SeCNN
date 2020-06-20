import hashlib
import collections
from operator import itemgetter
from pathlib import Path
import os

voca_path = 'data/vocabulary'



def makevoca(path, typ, counter):
    f = open('data/' + path + typ, 'r', encoding='utf-8')
    DATA = f.readlines()
    f.close()
    for data in DATA:
        for c in data.strip().split():
            counter[c] += 1


def getvoca(typ, counter):
    makevoca('valid/', typ, counter)
    makevoca('train/', typ, counter)


def start():
    if not os.path.isdir(voca_path):
        os.mkdir(voca_path)

    counter = collections.Counter()
    getvoca('nl', counter)

    sorted_word_to_cnt = sorted(counter.items(),
                                key=itemgetter(1),
                                reverse=True)
    sorted_word = [x[0] for x in sorted_word_to_cnt]
    sorted_word = ['<pad>', '<unk>', '<start>', '<end>'] + sorted_word
    if len(sorted_word) > 30000:
        sorted_word = sorted_word[:30000]

    f = open('data/vocabulary/nl', 'w', encoding='utf-8')
    for w in sorted_word:
        f.write(w + '\n')
    f.close()

    counter = collections.Counter()
    getvoca('sbt', counter)

    sorted_word_to_cnt = sorted(counter.items(),
                                key=itemgetter(1),
                                reverse=True)
    sorted_word = [x[0] for x in sorted_word_to_cnt]
    sorted_word = ['<pad>', '<unk>'] + sorted_word
    if len(sorted_word) > 30000:
        sorted_word = sorted_word[:30000]

    f = open('data/vocabulary/sbt', 'w', encoding='utf-8')
    for w in sorted_word:
        f.write(w + '\n')
    f.close()

    counter = collections.Counter()
    getvoca('code', counter)

    sorted_word_to_cnt = sorted(counter.items(),
                                key=itemgetter(1),
                                reverse=True)
    sorted_word = [x[0] for x in sorted_word_to_cnt]
    sorted_word = ['<pad>', '<unk>'] + sorted_word
    if len(sorted_word) > 30000:
        sorted_word = sorted_word[:30000]

    f = open('data/vocabulary/code', 'w', encoding='utf-8')
    for w in sorted_word:
        f.write(w + '\n')
    f.close()


