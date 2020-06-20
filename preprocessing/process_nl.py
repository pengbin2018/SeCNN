import os

def get_data(path):
    f = open('data/'+path+'/nl', 'r', encoding='utf-8')
    NL = f.readlines()
    f.close()
    os.remove('data/'+path+'/nl')
    f = open('data/' + path + '/nl', 'w', encoding='utf-8')

    for nl in NL:
        temp =nl.strip().split()
        s = ''
        for c in temp:
            if c =='.' or c =='?' or c ==';':
                s+=c
                break
            s+=c+' '
        temp = s
        s =''
        for c in temp:
            s+=c
        s+='\n'
        f.write(s)
    f.close()


def start():
    print('process test nl......')
    get_data('test')

    print('process train nl......')
    get_data('train')

    print('process valid nl......')
    get_data('valid')
