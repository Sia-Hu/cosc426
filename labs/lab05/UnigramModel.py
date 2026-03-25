import math
import csv
import nltk

class UnigramModel:
    def __init__(self, tokenize, tokenizer_kwargs: dict, vocab: set, unk_token:str, train_paths: list, smooth:str, lower:bool):
        self.tokenize = tokenize #tokenizing function
        self.tokenizer_kwargs = tokenizer_kwargs #args for tokenize function
        self.smooth = smooth
        self.vocab = vocab
        self.lower = lower  
        self.unk_token = unk_token

        self.traindat = self.preprocess(train_paths)
        self.unigram_freqs = self.train()


    def preprocess(self,fpaths):
        """
        Params: 
            fpaths: list of filepaths

        Returns:
            tokenized nested list of all words in all filepaths. Each sublist is a sequence.  

        """
        dat = []
        for fname in fpaths:
            with open(fname, 'r') as f:
                text = f.readlines()

            for sent in text: 
                if self.lower:
                    sent = sent.lower()

                # split text to list of tokens
                curr_dat = self.tokenize(sent, self.tokenizer_kwargs)

                # replace tokens not in vocab with unk token
                curr_dat = [word if word in self.vocab else self.unk_token for word in curr_dat]

                dat.append(curr_dat)

        return dat

    def train(self):
        """
        Returns:
            Dictionary with frequencies of all the unigrams in the trained model

        """
        freq_dict = {}
        for sent in self.traindat:
            for word in sent:
                freq_dict[word] = freq_dict.get(word, 0)+1


        return freq_dict


    def get_prob(self,unigram):
        """
        Params:
            unigram. Assumes unigram is preprocessed (e.g., words not in vocab are already replaced with UNK)
        Returns: 
            Smoothed probability of unigram given the trained model. -1.0 if invalid smooth. (Valid smooth:  MLE (no smoothing), add-k where you add k to all bigram counts)

        """
        if unigram not in self.vocab:
            unigram = self.unk_token
   
        if self.smooth == 'MLE':
            total_count = sum(self.unigram_freqs.values())
            word_count = self.unigram_freqs.get(unigram,0)
            return word_count / total_count if total_count > 0 else 0.0
        elif self.smooth.startswith('add-'):
            try:
                k = float(self.smooth.split('-')[1])
            except ValueError:
                return -1.0
            total_count = sum(self.unigram_freqs.values())
            vocab_size = len(self.vocab)
            word_count = self.unigram_freqs.get(unigram,0)
            return (word_count + k) / (total_count + k * vocab_size) if total_count > 0 else 0.0
        else:
            return -1.0



    def evaluate(self, datafpath, predfpath):
        """
        Params:
            datafpath: path to the file with the data
            predfpath: path where predictions will be stored

        Returns: 
            Nothing. But creates a file in predfpath with the following columns: 
                sentid: id of the sentence; assume each sentence is on a new line
                word: current word
                wordpos: position of the word in the sentence
                prob: P(word | context)
                surp: -log_2(word | context)
        """
        results = []
        testdat = self.preprocess([datafpath])
        for sentid, sent in enumerate(testdat):
            for wordpos, word in enumerate(sent):
                prob = self.get_prob(word)
                if prob > 0:
                    surp = -math.log2(prob)
                else:
                    surp = float('inf')
                results.append({'sentid': sentid, 'word': word, 'wordpos': wordpos, 'prob': prob, 'surp': surp})
        
        with open(predfpath, 'w', newline='') as csvfile:
            fieldnames = ['sentid', 'word', 'wordpos', 'prob', 'surp']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames, delimiter='\t')
            writer.writeheader()
            for row in results:
                writer.writerow(row)
        return