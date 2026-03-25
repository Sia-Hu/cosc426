import nltk
import math
from transformers import AutoTokenizer

def get_ngrams(text:list, n):
    """
    Params:
        text: tokenized text in a list
        n: the n of the ngram

    Returns:
        list of all ngrams
    """
    return list(nltk.ngrams(text, n))

def get_ngramFreqs(text: list, n: int):
    """
    Params:
        text: text, split into list of tokens 
        n: the n for ngram

    Returns:
        Frequnency dictionary
    """

    ngrams = get_ngrams(text, n)
    freq_dict = nltk.probability.FreqDist(ngrams)

    return freq_dict

def preprocess(textfname: list, lower, tokenizer, **kwargs):
    """
    Params:
        textfname: path to text file. 
        tokenizer: tokenizing function 
        **kwargs: other kwargs for the tokenizer

    Returns: 
        List of tokens in the text

    """
    tokens = []
    print(f'Reading {textfname}')
    with open(textfname, 'r') as f:
        text = f.readlines()

    for i,line in enumerate(text):
        if lower:
            line = line.lower()
        tokens.extend(tokenizer(line, kwargs))
        if i%100 == 0:
            print(f'Tokenized {i+1} lines')

    return tokens

def hf_tokenize(text:str, kwargs):
    """
    Params: 
        text: string of text
        kwargs: dictionary with kwargs. Should include key 'modelname' which specifies the hf modelname
    Returns: 


    """
    modelname = kwargs['modelname']

    tokenizer = AutoTokenizer.from_pretrained(modelname)
    tokenized_output = tokenizer(text)
    words = tokenizer.convert_ids_to_tokens(tokenized_output['input_ids'])
    return words

def get_hf_vocab(modelname):
    """
    Params:
        modelname: string of hf modelname
    Returns:
        The vocabulary used by the huggingface model 

    """
    tokenizer = AutoTokenizer.from_pretrained(modelname)
    return tokenizer.vocab

def get_ngram_prob(ngram: tuple, smooth: str, ngram_freqs: dict, n_minus1_freqs: dict, vocab_size: int):
    
    if smooth == 'MLE':
        count_ngram = ngram_freqs.get(ngram, 0)
        n_minus1 = ngram[:-1]
        count_n_minus1 = n_minus1_freqs.get(n_minus1, 0)
        if count_n_minus1 == 0:
            return 0.0
        else:
            return count_ngram / count_n_minus1
    elif smooth.startswith('add-'):
        try:
            k = float(smooth.split('-')[1])
        except ValueError:
            return -1.0
        count_ngram = ngram_freqs.get(ngram, 0)
        n_minus1 = ngram[:-1]
        count_n_minus1 = n_minus1_freqs.get(n_minus1, 0)
        return (count_ngram + k) / (count_n_minus1 + k * vocab_size)
    else:
        return -1.0

def evaluate(text: list, smooth: str, n: int, ngram_freqs: dict, n_minus1_freqs: dict, vocab_size: int, ):

    log_prob_sum = 0.0
    ngrams = get_ngrams(text, n)
    for ngram in ngrams:
        prob = get_ngram_prob(ngram, smooth, ngram_freqs, n_minus1_freqs, vocab_size)
        if prob > 0:
            log_prob_sum += math.log(prob)
        else:
            log_prob_sum += math.log(1e-10)  
            
    avg_log_prob = log_prob_sum / len(ngrams)
    perplexity = math.exp(-avg_log_prob)
    return perplexity


def tests():
    
    text = preprocess('data/test.txt', True, hf_tokenize, modelname='distilgpt2')

    bigram_freqs = get_ngramFreqs(text, 2)

    bigrams = get_ngrams(text, 2)

    for bigram in bigrams:
        if bigram_freqs[bigram] !=1:
            print(bigram, bigram_freqs[bigram])

    print(len(get_hf_vocab('distilgpt2')))

if __name__ == "__main__":
    tests()