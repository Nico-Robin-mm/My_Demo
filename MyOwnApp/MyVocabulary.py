import numpy as np
# from collections import defaultdict


class Vocab:
    """build my own vocabulary test book"""
    def __init__(self):
        self.word2idx = dict()
        self.idx2word = dict()
        self.translation = dict()
        self.idx = 0
        
        # saved path
        self.savefile = r"myvocab\myvocab.txt"
        
        # loaded path
        self.loadfile = r"myvocab\生词.txt"
    
    def add(self, word, meaning):
        assert word not in self.translation, f"the word '{word}' exists!"
        self.idx = max(self.idx2word) + 1
        self.word2idx[word] = self.idx
        self.idx2word[self.idx] = word
        self.translation[word] = [meaning]
        self.idx += 1
        
    @property
    def size(self):
        return len(self.translation)
    
    def save(self):
        with open(self.savefile, "w") as f:
            for i in self.idx2word:
                f.write(f"{i:<5}{self.idx2word[i]}\t{self.translation[self.idx2word[i]]}\n")
                
    def load(self):
        with open(self.loadfile, "r", encoding="utf-8") as f:
            words = f.readlines()
            for word in words:
                idx, word, *meaning = word.strip().split(maxsplit=2)
                idx = int(idx)
                self.idx2word[idx] = word
                self.word2idx[word] = idx
                self.translation[word] = meaning[0]
            self.idx = len(words)
    
    def delete(self, word):
        assert word in self.translation, "no such word!"
        del self.idx2word[self.word2idx[word]]
        del self.word2idx[word]
        del self.translation[word]
    
    # rearrange the idx of vocabulary
    def update(self):
        self.idx = len(self.translation)
        self.idx2word = dict()
        self.word2idx = dict()
        for idx, word in enumerate(self.translation):
            self.idx2word[idx] = word
            self.word2idx[word] = idx
    
    # select {num} words to test
    def test(self, num=1):
        tmp = self._test()
        for idx in range(num):
            print(tmp.__next__())
        
    def _test(self):
        re_idx = np.random.permutation(len(self.translation))
        for idx in re_idx:
            yield self.idx2word[idx]
        
        
if __name__ == "__main__":
    my = Vocab()
    my.load()
    # my.save()
