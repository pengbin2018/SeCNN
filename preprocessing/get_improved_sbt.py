import re
import shutil,os


def Tran(test):
    return re.compile(r'([a-z]|\d)([A-Z])').sub(r'\1_\2', test).lower()


def sptNODE(node, p):
    temp = node.split('.')
    mysbt = []
    pos = []
    for tp in temp:
        tp = Tran(tp)
        w = tp.split('_')
        for t in w:
            mysbt.append(t)
            pos.append(p)
    return mysbt, pos


def dfsSBT(sbt, mysbt, pos, i, p):
    mysbt.append(sbt[i + 1])
    nw = sbt[i + 1]
    pos.append(p)
    no = p
    p += 1
    i += 2
    while True:
        if sbt[i] == ')':
            if nw == sbt[i + 1]:
                mysbt.append(nw)
                pos.append(no)
            else:
                tem = sptNODE(sbt[i + 1], no)
                mysbt += tem[0]
                pos += tem[1]
            break
        i, p = dfsSBT(sbt, mysbt, pos, i, p)

    return i + 2, p


def mySBT(sbt):
    temp = sbt.split()
    mysbt = []
    pos = []
    dfsSBT(temp, mysbt, pos, 0, 0)
    s1 = ''
    s2 = ''
    for st, sp in zip(mysbt, pos):
        if st == '':
            continue
        s1 += st + ' '
        s2 += str(sp) + ' '
    return s1+'\n', s2+'\n'


def get_data(path):
    f = open('data/' + path + '/sbt', 'r', encoding='utf-8')
    SBT = f.readlines()
    f.close()
    os.remove('data/' + path + '/sbt')

    f1 = open('data/' + path + '/sbt', 'w', encoding='utf-8')
    f2 = open('data/' + path + '/pos', 'w', encoding='utf-8')
    for sbt in SBT:
        temp = mySBT(sbt)
        f1.write(temp[0])
        f2.write(temp[1])
    f1.close()
    f2.close()

def start():
    print('generate test improved sbt......')
    get_data('test')

    print('generate train improved sbt......')
    get_data('train')

    print('generate valid improved sbt......')
    get_data('valid')
