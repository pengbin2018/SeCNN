f = open('data/train/sbt', 'r', encoding='utf-8')
SBT = f.readlines()
f.close()

f = open('data/train/pos', 'r', encoding='utf-8')
POS = f.readlines()
f.close()
for sbt, pos in zip(SBT, POS):
    print(len(sbt.split()), len(pos.split()))