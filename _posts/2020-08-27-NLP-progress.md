---
published: true
layout: post
title: NLP progress (brainstorming)
permalink: /nlp-progress
---
- [NLP processing](#nlp-processing)
- [BOW](#bow)
- [Co-occurrence matrix](#co-occurrence-matrix)
- [word2vec](#word2vec)
- [GloVe (Global Vectors) for Word Representation)](#glove-global-vectors-for-word-representation)
- [The Transformer model](#the-transformer-model)
- [Tokenizers](#tokenizers)
- [NLP libraries](#nlp-libraries)

## NLP processing

NLP processing started with regular expressions and with the text normalization methods such as:
* tokenization
* convert words to lower case
* lemmatization 
* stemming

**Tokenization** is splitting sentences into words with optional removal of stopwords, punctuation, words with less than 3 characters, etc.

__Lemmatization__ assumes morphological word analysis to return the base  form of a word, while stemming is more brute removal of the word endings or affixes in general.


First the statistical NLP methods came along together with the model called Bag Of Words (BOW).

## BOW

The __BOW__ idea is to group words together where the order of words is not important.

Known statistical methods based on BOW are LSA or Latent Statistical Analysis
and LDA or Latent Dirichlet Analysis..

With LSA the idea is that if words have similar meaning they will appear in similar text contexts.

It was important because it brought the notion of word similarities. We could later project the words in 2D and get the better word understanding based on similarities.

With LDA you could ask to detect different text categories from your corpora and it could provide the keywords for each category. LDA is actually dimensionality reduction technique.


## Co-occurrence matrix

Needed for many techniques is the co-occurrence matrix. The idea is to create a matrix based on text (distinct words). 

For instance with this text:

```
He is a clever boy.
She is a clever girl.
```
We can create the matrix of co-occurrence:

|% | He | is | a  |clever| boy | she | girl
|-- |  -- | -- | -- | -- | --  | -- | -- |
|He |  | 1 |  |  |   |  |  | 
|is |  |  | 2 |  |   |  |  | 
|a |  |  |  | 2 |   |  |  | 
|clever |  |  |  |  | 1  |  | 1 | 
|boy |  |  |  |  |   |  |  | 
|she |  | 1 |  |  |   |  |  | 
|girl |  |  |  |  |   |  |  |

If the text is big we can get big matrices, typically we could limit that to 10,000 words, because we can ignore words we don't care about. However, if we would like to deal with 170,000 English language has the matrix will be very big.

The upper matrix has been produced with teh Markov assumption in mind (pay attention on just the next word), but we can easily make the sliding windows bigger and in both directions. 

This is how the co-occurrence matrix will become symmetric, but easily we can create the symmetric matrix in numpy with this trick:

```
M = M + M.T
```

If we would like to get word projections in 2D or any kind of analysis we would need to lower the space so we need to use some PCA and in general SVD technique.

This would lower the dimension of our matrix from `NxN` to `NxK` where `K<<N`

Here is one example, for the text:
```
He is a king. 
She is a queen.
He is a man. 
She is a woman.
Berlin is germany capital.
Paris is france capital.
```
Find the similar words:

![similar words](/images/projection2d.png)


## word2vec
Then in 2013 one very important algorithm `word2vec` came along where.

Using this algorithm it was possible to learn the representation of words where each word had at least 50 or up to 1000 latent features.

The major gain with this latent approach, we are not forced to create matrices of NxN dimension where N is the number of distinct words. Instead all we have to learn is the NxK matrix where K is usually close to 100 (from 50 till 1000 usually).

This break-trough idea [published by Mikolov et al.](https://arxiv.org/abs/1301.3781){:rel="nofollow"} was capable of doing word arithmetics.

```
W['king'] - W['man'] + W['woman'] = W['queen'] 
```

This provided a mean to deal with word analogies, because you could extend this idea to anything, but you could also understand the bias word model or language in general may have.

`word2vec` uses n-grams. Here are some possible 3-gram for the text: 

> He is a king. She is a queen.
```
[(['he', 'is'], 'a'),
 (['is', 'a'], 'king'),
 (['a', 'king'], 'she'),
 (['king', 'she'], 'is'),
 (['she', 'is'], 'a'),
 (['is', 'a'], 'queen')]
 ```

## GloVe (Global Vectors) for Word Representation)

[GloVe paper](https://nlp.stanford.edu/pubs/glove.pdf){:rel="nofollow"} was another step forward in NLP.

It uses the context window and matrix factorization tricks like we would use used for **big** co-occurrence matrix.

The difference is; instead of measuring word frequencies directly we take the `log` co-occurrence counts which improves the impact of no so frequent words.

The paper also solved the problem when the co-occurrence count overshouts the maximal allowed number defined by the data format; in this case the number stays constant.


## The Transformer model

In the 2017 the paper [Attention Is All You Need](https://arxiv.org/abs/1706.03762){:rel="nofollow"} changed the horizon of machine learning and NLP in general.

It introduced the attention to NLP. The concept is based on vector dot product, parameters called __Query__, **Key** and **Value** parameters and **softmax** function that extract he most probable combination of tokens.

It almost completely took the NLP world which at that time continuously made progress with LSTM, GRU and other recursive models that are sequential in nature.

With models that are sequential in nature you convert word to tokens first and then you process tokens sequentially trough the model.

Transformer model processes the great number of tokens from once (512, 1024, or even greater) limited just with the amount of available memory and model definition).

From the transformer model many new concepts started to grow. For instance, the famous BERT model would be the transformer where we removed the decoder part.

Nowadays a great attention is on GPT-2 and GPT-3 models and their modifications. 

GPT based models can generate high quality text based on the initial inputs.


In general Transformers models can do all kind of [NLP tasks](https://dejanbatanjac.github.io/nlp-acronyms/).

## Tokenizers 

The key parts of the transformers are the so called tokenizers. Very popular tokenizers today are:

* Spacy
* WordPeace
* SentencePeace
* BPT

Here are models that use famous [tokenizers](https://github.com/huggingface/tokenizers){:rel="nofollow"}:

* WordPeace: BERT, DistilBERT, Electra
* BPT: GPT-2, Roberta
* SentencePeace: T5, ALBERT, CamemBERT, XLMRoBERTa, XLNet, Marian
* Spacy: GPT


The great idea of **custom tokenizer** is that you can train the tokenizer to much your text corpora. [Transformers library](https://github.com/huggingface/transformers){:rel="nofollow"} provides such tokenizers and training howto examples.

[Transformers library](https://huggingface.co/){:rel="nofollow"} is all about customizing transformer based models and achieving **transfer learning**, and of course you are not meant just to customize the models or train them from scratch, you can use the already pretrained modes.

## NLP libraries

There is a new [Transformers library for NLP](https://github.com/huggingface/nlp){:rel="nofollow"}.

The older good NLP libraries still around you may use for many NLP tasks:


* nltk (Natural Language Toolkit)
* word2vec
* glove
* gensim

_Example_: Create tokens with nltk:

```python
import nltk
nltk.download('punkt')

# import nltk.data
tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

from nltk.tokenize import word_tokenize

text = """He is a king. 
She is a queen.
He is a man. 
She is a woman.
Berlin is germany capital.
Paris is france capital."""
lists = nltk.tokenize.sent_tokenize(text)

short=[]
for sentence in lists:    
    words= word_tokenize(sentence.lower())
    short.append(words)   
    
pprint.pprint(short)
```
_Output_:
```
[['he', 'is', 'a', 'king', '.'],
 ['she', 'is', 'a', 'queen', '.'],
 ['he', 'is', 'a', 'man', '.'],
 ['she', 'is', 'a', 'woman', '.'],
 ['berlin', 'is', 'germany', 'capital', '.'],
 ['paris', 'is', 'france', 'capital', '.']]
```


_Example_: Find synonyms and hypenyms

```python
import nltk
nltk.download('wordnet')
from nltk.corpus import wordnet as wn

# synonym set
for synset in wn.synsets('flush'):
    for l in synset.lemmas():
        print(l.name(), synset.pos() , synset)
        
# dog hypernyms
dog = wn.synset('dog.n.01')
list(dog.closure(lambda s: s.hypernyms()))
```
_Output_:
```
flower n Synset('flower.n.03')
prime n Synset('flower.n.03')
peak n Synset('flower.n.03')
heyday n Synset('flower.n.03')
bloom n Synset('flower.n.03')
blossom n Synset('flower.n.03')
efflorescence n Synset('flower.n.03')
flush n Synset('flower.n.03')
bloom n Synset('bloom.n.04')
blush n Synset('bloom.n.04')
flush n Synset('bloom.n.04')
rosiness n Synset('bloom.n.04')
hot_flash n Synset('hot_flash.n.01')
flush n Synset('hot_flash.n.01')
flush n Synset('flush.n.04')
bang n Synset('bang.n.04')
boot n Synset('bang.n.04')
charge n Synset('bang.n.04')
rush n Synset('bang.n.04')
flush n Synset('bang.n.04')
thrill n Synset('bang.n.04')
kick n Synset('bang.n.04')
flush n Synset('flush.n.06')
gush n Synset('flush.n.06')
outpouring n Synset('flush.n.06')
blush n Synset('blush.n.02')
flush n Synset('blush.n.02')
blush v Synset('blush.v.01')
crimson v Synset('blush.v.01')
flush v Synset('blush.v.01')
redden v Synset('blush.v.01')
flush v Synset('flush.v.02')
flush v Synset('flush.v.03')
flush v Synset('flush.v.04')
level v Synset('flush.v.04')
even_out v Synset('flush.v.04')
even v Synset('flush.v.04')
flush v Synset('flush.v.05')
scour v Synset('flush.v.05')
purge v Synset('flush.v.05')
sluice v Synset('sluice.v.02')
flush v Synset('sluice.v.02')
flush v Synset('flush.v.07')
flush s Synset('flush.s.01')
affluent s Synset('affluent.s.01')
flush s Synset('affluent.s.01')
loaded s Synset('affluent.s.01')
moneyed s Synset('affluent.s.01')
wealthy s Synset('affluent.s.01')
flush r Synset('flush.r.01')
flush r Synset('flush.r.02')


[Synset('canine.n.02'),
 Synset('domestic_animal.n.01'),
 Synset('carnivore.n.01'),
 Synset('animal.n.01'),
 Synset('placental.n.01'),
 Synset('organism.n.01'),
 Synset('mammal.n.01'),
 Synset('living_thing.n.01'),
 Synset('vertebrate.n.01'),
 Synset('whole.n.02'),
 Synset('chordate.n.01'),
 Synset('object.n.01'),
 Synset('physical_entity.n.01'),
 Synset('entity.n.01')]
```



