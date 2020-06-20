import re
import os


def Tran(test):
    return re.compile(r'([a-z]|\d)([A-Z])').sub(r'\1_\2', test).lower()


def mysplit(words):
    temp = words.strip().split()
    s = ''
    for w in temp:
        if w == '_STR' or w == '_NUM' or w == '_BOOL':
            s +=w+' '
            continue
        w1 = Tran(w)
        w2 = w1.split('_')
        for w3 in w2:
            s+= w3+' '
    return s+'\n'

def getData(path):
    f = open('data/'+path+'/code', 'r', encoding='utf-8')
    CODE= f.readlines()
    f.close()
    os.remove('data/'+path+'/code')
    f = open('data/'+path+'/code', 'w', encoding='utf-8')
    for code in CODE:
        s = mysplit(code)
        f.write(s)
    f.close()

def start():
    print('process test code......')
    getData('test')

    print('process train code......')
    getData('train')

    print('process valid code......')
    getData('valid')
