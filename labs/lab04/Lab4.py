import math
import doctest
import nltk
nltk.download('punkt')

def getVocab(vocab_fname: str) -> set: 
    """
    Args: 
        vocab_fname: filepath to vocab file. Each line has a new vocab item

    Returns: 
        A set of all the vocabulary items in the file plus three additional tokens: 
            - [UNK] : to represent words in the text not in the vocab
            - [BOS] : to represent the beginning of sentences. 
            - [EOS] : to represent the end of sentences. 
        If you run this function on the glove vocab, it should return set with 400003 items.

    >>> len(getVocab('data/glove_vocab.txt'))
    400003
    """
    file = open(vocab_fname, 'r', encoding='utf-8')
    vocab = set()
    for line in file:
        vocab.add(line.strip())
    vocab.add('[UNK]')
    vocab.add('[BOS]')
    vocab.add('[EOS]')
    return vocab
    pass

def preprocess(textfname:str, mark_ends: bool) -> list:
    """
    Args: 
        text: some text

        mark_ends: indicates whether sentences should start with [BOS] and end with [EOS]

    Returns: 
        A list of lists where each sublist consists of tokens from each sentence. 
        Use existing nltk functions to first divide the text into sentences, and then into words. 

    >>> preprocess('data/test.txt', mark_ends=True)
    [['[BOS]', 'one', 'thing', 'was', 'certain', ',', 'that', 'the', '_white_', 'kitten', 'had', 'had', 'nothing', 'to', 'do', 'with', 'it', ':', '—it', 'was', 'the', 'black', 'kitten', '’', 's', 'fault', 'entirely', '.', '[EOS]'], ['[BOS]', 'for', 'the', 'white', 'kitten', 'had', 'been', 'having', 'its', 'face', 'washed', 'by', 'the', 'old', 'cat', 'for', 'the', 'last', 'quarter', 'of', 'an', 'hour', '(', 'and', 'bearing', 'it', 'pretty', 'well', ',', 'considering', ')', ';', 'so', 'you', 'see', 'that', 'it', '_couldn', '’', 't_', 'have', 'had', 'any', 'hand', 'in', 'the', 'mischief', '.', '[EOS]']]

    >>> preprocess('data/test.txt', mark_ends=False)
    [['one', 'thing', 'was', 'certain', ',', 'that', 'the', '_white_', 'kitten', 'had', 'had', 'nothing', 'to', 'do', 'with', 'it', ':', '—it', 'was', 'the', 'black', 'kitten', '’', 's', 'fault', 'entirely', '.'], ['for', 'the', 'white', 'kitten', 'had', 'been', 'having', 'its', 'face', 'washed', 'by', 'the', 'old', 'cat', 'for', 'the', 'last', 'quarter', 'of', 'an', 'hour', '(', 'and', 'bearing', 'it', 'pretty', 'well', ',', 'considering', ')', ';', 'so', 'you', 'see', 'that', 'it', '_couldn', '’', 't_', 'have', 'had', 'any', 'hand', 'in', 'the', 'mischief', '.']]

    """
    file = open(textfname, 'r', encoding='utf-8')
    text = file.read()
    sentences = nltk.sent_tokenize(text)
    tokenized_sentences = []
    for sentence in sentences:
        tokens = nltk.word_tokenize(sentence)
        # change the first token to be lowercase
        tokens[0] = tokens[0].lower()
        if mark_ends:
            tokens = ['[BOS]'] + tokens + ['[EOS]']
        tokenized_sentences.append(tokens)
    return tokenized_sentences
    pass


def TestBigramFreqs(freq_dict, print_non1 = False):
    """
    Helper function to use in doctest of getBigramFreqs

    """
    inverse = {}
    # inverse2 = {}
    for key,val in freq_dict.items():
        # inverse[val] = inverse.get(val, 0) + 1
        inverse[val] = inverse.get(val, []) 
        inverse[val].append(key)

    if print_non1:
        return {key: val for key, val in inverse.items() if key !=1}
    else:
        return {key: len(val) for key, val in inverse.items()}

def getBigramFreqs(preprocessed_text:list, vocab:set) -> dict:
    """
    Args: 
        preprocessed_text: text that has been divided into sentences and tokens


    Returns: 
        dictionary with all bigrams that occur in the text along with frequencies. 
        Each key should be a tuple of strings of the format (first_token, second_token). 

    >>> TestBigramFreqs(getBigramFreqs(preprocess('data/test.txt', mark_ends=True), getVocab('data/glove_vocab.txt')))
    {1: 70, 2: 3}

    >>> TestBigramFreqs(getBigramFreqs(preprocess('data/test.txt', mark_ends=True), getVocab('data/glove_vocab.txt')), print_non1=True)
    {2: [('kitten', 'had'), ('.', '[EOS]'), ('for', 'the')]}

    """
    bigram ={}
    for sentence in preprocessed_text:
        for i in range(len(sentence)-1):
            first = sentence[i] if sentence[i] in vocab else '[UNK]'
            second = sentence[i+1] if sentence[i+1] in vocab else '[UNK]'
            pair = (first, second)
            bigram[pair] = bigram.get(pair, 0) + 1
    return bigram
    pass

def getUnigramFreqs(preprocessed_text:list, vocab:set) -> dict:
    unigram = {}
    for sentence in preprocessed_text:
        for token in sentence:
            word = token if token in vocab else '[UNK]'
            unigram[word] = unigram.get(word, 0) + 1
    return unigram

def getBigramProb(bigram: tuple, smooth: str, **kwargs):
    """
    Args:
        bigram: the tuple of the bigram you want the prob of
        smooth: MLE (no smoothing), add-k where you add k to all bigram counts. Returns -1 if invalid smooth is entered. 
        **kwargs: other parameters you might want. 

        Hint: think about what parameters do you want to pass in so you minimize redundant computation. 

    Returns:
        float with prob. 
        Return -1.0 if invalid smoothing value is entered. 

    >>> getBigramProb(('one', 'thing'), 'MLE', bigram_freqs=getBigramFreqs(preprocess('data/test.txt', mark_ends=True),getVocab('data/glove_vocab.txt')), unigram_freqs=getUnigramFreqs(preprocess('data/test.txt', mark_ends=True), getVocab('data/glove_vocab.txt')), vocab_size=len(getVocab('data/glove_vocab.txt')))
    1.0
    >>> getBigramProb(('one', 'thing'), 'add-1', bigram_freqs=getBigramFreqs(preprocess('data/test.txt', mark_ends=True),getVocab('data/glove_vocab.txt')), unigram_freqs=getUnigramFreqs(preprocess('data/test.txt', mark_ends=True), getVocab('data/glove_vocab.txt')), vocab_size=len(getVocab('data/glove_vocab.txt')))
    4.999950000499995e-06
    """   
    bigram_freqs = kwargs.get('bigram_freqs', {})
    unigram_freqs = kwargs.get('unigram_freqs', {})
    vocab_size = kwargs.get('vocab_size', 1)
    if smooth == 'MLE':
        (first, second) = bigram
        bigram_count = bigram_freqs.get(bigram, 0)
        unigram_count = unigram_freqs.get(first, 0) 
        if unigram_count == 0:
            return 0.0
        return bigram_count / unigram_count
    elif smooth.startswith('add-'):
        try:
            k = float(smooth.split('-')[1])
        except ValueError:
            return -1.0
        (first, second) = bigram
        bigram_count = bigram_freqs.get(bigram, 0)
        unigram_count = unigram_freqs.get(first, 0) 
        return (bigram_count + k) / (unigram_count + k * vocab_size)
    else:
        return -1.0 
    pass

def evaluateBigramModel(test_text: str, bigram_freqs: dict, unigram_freqs: dict, vocab: set, smooth: str = 'add-1', mark_ends : bool = True) -> float:
    preprocessed_test = preprocess(test_text, mark_ends=mark_ends)
    
    log_prob_sum = 0.0
    total_bigrams = 0
    
    vocab_size = len(vocab)
    
    for sentence in preprocessed_test:
        for i in range(len(sentence) - 1):
            first = sentence[i] if sentence[i] in vocab else '[UNK]'
            second = sentence[i + 1] if sentence[i + 1] in vocab else '[UNK]'
            bigram = (first, second)
            
            prob = getBigramProb(bigram, smooth, 
                               bigram_freqs=bigram_freqs,
                               unigram_freqs=unigram_freqs,
                               vocab_size=vocab_size)
            
            if prob > 0:
                log_prob_sum += math.log(prob)
            else:
                log_prob_sum += math.log(1e-10)  
                
            total_bigrams += 1
    
    avg_log_prob = log_prob_sum / total_bigrams
    perplexity = math.exp(-avg_log_prob)
    
    return perplexity