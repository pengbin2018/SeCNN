

class Tokenization:
    def __init__(self):
        f = open('data/vocabulary/sbt', 'r', encoding='utf-8')
        sbt_voca = f.readlines()
        f.close()
        self.sbt_word_to_id = {}
        self.sbt_id_to_word = {}
        for i, w in enumerate(sbt_voca):
            w = w.strip()
            self.sbt_id_to_word[i] = w
            self.sbt_word_to_id[w] = i

        f = open('data/vocabulary/code', 'r', encoding='utf-8')
        code_voca = f.readlines()
        f.close()
        self.code_word_to_id = {}
        self.code_id_to_word = {}
        for i, w in enumerate(code_voca):
            w = w.strip()
            self.code_id_to_word[i] = w
            self.code_word_to_id[w] = i

        f = open('data/vocabulary/nl', 'r', encoding='utf-8')
        nl_voca = f.readlines()
        f.close()
        self.nl_word_to_id = {}
        self.nl_id_to_word = {}
        for i, w in enumerate(nl_voca):
            w = w.strip()
            self.nl_id_to_word[i] = w
            self.nl_word_to_id[w] = i

    def sbt_id(self, word):
        return self.sbt_word_to_id[word] if word in self.sbt_word_to_id.keys() else self.sbt_word_to_id['<unk>']

    def sbt_word(self, id):
        return self.sbt_id_to_word[id] if id in self.sbt_id_to_word.keys() else '<unk>'

    def code_id(self, word):
        return self.code_word_to_id[word] if word in self.code_word_to_id.keys() else self.code_word_to_id['<unk>']

    def code_word(self, id):
        return self.code_id_to_word[id] if id in self.code_id_to_word.keys() else '<unk>'

    def nl_id(self, word):
        return self.nl_word_to_id[word] if word in self.nl_word_to_id.keys() else self.nl_word_to_id['<unk>']

    def nl_word(self, id):
        return self.nl_id_to_word[id] if id in self.nl_id_to_word.keys() else '<unk>'

