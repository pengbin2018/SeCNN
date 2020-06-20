
sbt_leng = 300
code_leng = 200
nl_leng = 30

def get_data(bacth, path, tokens):
    sbts = []
    sbtposs = []
    sbtLeng = []

    codes = []
    codeLeng = []

    inputNL = []
    outputNL = []
    nlLeng = []
    nlopuputLeng = []
    f = open('data/' +path+ '/sbt', 'r', encoding='utf-8')
    SBTs = f.readlines()
    f.close()
    for temp in SBTs:
        sbt = [tokens.sbt_id(w) for w in temp.strip().split()]
        if len(sbt) > sbt_leng:
            sbt = sbt[0:sbt_leng]
        while len(sbt) < sbt_leng:
            sbt.append(0)
        sbts.append(sbt)
        sbtLeng.append(sbt_leng)

    f = open('data/' +path+ '/pos', 'r', encoding='utf-8')
    SBTs = f.readlines()
    f.close()
    for temp in SBTs:
        sbt = [int(w) for w in temp.strip().split()]
        if len(sbt) > sbt_leng:
            sbt = sbt[0:sbt_leng]
        while len(sbt) < sbt_leng:
            sbt.append(0)
        sbtposs.append(sbt)

    f = open('data/' +path+ '/code', 'r', encoding='utf-8')
    CODEs = f.readlines()
    f.close()
    for temp in CODEs:
        code = [tokens.code_id(w) for w in temp.strip().split()]
        if len(code) > code_leng:
            code = code[0:code_leng]
        while len(code) < code_leng:
            code.append(0)
        codes.append(code)
        codeLeng.append(code_leng)


    f = open('data/' +path+ '/nl', 'r', encoding='utf-8')
    NLs = f.readlines()
    f.close()
    if path == 'train':
        for temp in NLs:
            nl = [tokens.nl_id(w) for w in temp.strip().split()]
            nl = [2] + nl + [3]
            inp = nl[0:-1]
            outp = nl[1:len(nl)]
            if len(inp) > nl_leng:
                inp = inp[0:nl_leng]
                outp = outp[0:nl_leng]
            nlopuputLeng.append(len(inp))
            nlLeng.append(nl_leng)
            while len(inp) < nl_leng:
                inp.append(0)
                outp.append(0)
            inputNL.append(inp)
            outputNL.append(outp)
    else:
        for temp in NLs:
            nl = temp.strip().split()
            if len(nl) > nl_leng:
                nl = nl[:nl_leng]
            outputNL.append(nl)
            inputNL.append([])
            nlopuputLeng.append(0)
            nlLeng.append(0)
    bacthsbt = []
    bacthsbtpos = []
    bacthsbtLeng = []
    bacthcode = []
    bacthcodeLeng = []
    bacthinputNL = []
    bacthoutputNL=[]
    bacthnlLeng = []
    bacthnloutputLeng = []
    start = 0
    while start < len(codes):
        end = min(start+bacth, len(codes))
        bacthsbt.append(sbts[start:end])
        bacthsbtpos.append(sbtposs[start:end])
        bacthsbtLeng.append(sbtLeng[start:end])

        bacthcode.append(codes[start:end])
        bacthcodeLeng.append(codeLeng[start:end])
        bacthinputNL.append(inputNL[start:end])
        bacthoutputNL.append(outputNL[start:end])
        bacthnlLeng.append(nlLeng[start:end])
        bacthnloutputLeng.append(nlopuputLeng[start:end])
        start+=bacth
    return bacthsbt, bacthsbtpos, bacthsbtLeng, bacthcode, bacthcodeLeng, bacthinputNL, bacthoutputNL, bacthnlLeng, bacthnloutputLeng