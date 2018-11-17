import re
import os
import MeCab
import shutil
from pathlib import Path
import subprocess
try:
    import sentencepiece as spm
except:
    print('WARNING: you cannot use sentencepiece model')
    
class Tokenizer(object):
    def __init__(self, tag):
        self.tag = tag
    
    def tokenize(self, text):
        return token_list

class MecabTokenizer(Tokenizer):
    def __init__(self, tag):
        self.tag = tag
        self.tagger = MeCab.Tagger(' ')
        self.tagger.parse('')
        
    def __getstate__(self):
        return self.tag
    
    def __setstate__(self, state):
        self.tag = state
        self.tagger = MeCab.Tagger(' ')
        self.tagger.parse('')

    def iter_surface(self, text):
        morph_list = self.tagger.parseToNode(text)
        while morph_list:
            token = morph_list.surface
            if len(token) > 0:
                yield token
            morph_list = morph_list.next

    def tokenize(self, text):
        return list(self.iter_surface(text))

class SPTokenizer(Tokenizer):
    def __init__(self, model_path):
        self.model_path = Path(model_path)
        self.tag = self.model_path.name
        self.spm = None
    
    def __getstate__(self):
        d = self.__dict__
        d["spm"] = None
        return d
        
    def tokenize(self, text):
        self.spm = self.spm or self.load_spm(self.model_path)
        return self.spm.EncodeAsPieces(text)
    
    @staticmethod
    def load_spm(model_path):
        sp = spm.SentencePieceProcessor()
        path = Path(model_path, 'model.model')
        assert path.exists(), 'spm model {} does not exist.'.format(path)
        sp.load(str(path))
        assert spm, 'model load failure. '+str(path)
        return sp
    
    @staticmethod
    def train(corpus_path, save_path, vocab_size, overwrite=False):
        assert overwrite or not Path(save_path, 'model.model').exists(), 'Model already exists'
        save_path = str(Path(save_path).resolve())
        corpus_path  = str(Path(corpus_path).resolve())
        os.makedirs(save_path, exist_ok=True)
        try:
            cp = subprocess.run(
                'spm_train --input={} --model_prefix=model --vocab_size={}'.format(corpus_path, vocab_size), 
                shell=True, 
                cwd=save_path, 
                stdout=subprocess.PIPE, 
                stderr=subprocess.PIPE
            )
            cp.check_returncode()
        except:
            print(cp.stdout.decode())
            print(cp.stderr.decode())
            shutil.rmtree(save_path)