---
published: false
layout: post
title: Train hugging face models from scratch and fine tune them on various downstream tasks.
permalink: /transformers-train
---
- [Tokenizers](#tokenizers)
  - [Train your own tokenizer](#train-your-own-tokenizer)
- [Save the Tokenizer](#save-the-tokenizer)
- [Models](#models)
- [Save Model](#save-model)

To train [Tansformers](https://github.com/huggingface/transformers){:rel="nofollow"} from HuggingFace you need untrained models and the training process. Of course, you need to have the tokenizers. 

## Tokenizers

Many tokenizers exist but HuggingFace has [Tokenizerss](https://github.com/huggingface/tokenizers/tree/master/bindings/python){:rel="nofollow"}:

There are several generic tokenizers in there:
* CharBPETokenizer: The original BPE
* ByteLevelBPETokenizer: The byte level version of the BPE
* SentencePieceBPETokenizer: A BPE implementation compatible with the one used by SentencePiece
* BertWordPieceTokenizer: (original Bert tokenizer, using WordPiece)

But we can create custom tokenizers as well just for the specific text. This is relatively new idea that the tokenizer should best fit to the text. For instance tokenizer for Esperanto may not be the same as for the plain English text. In order to have and use the custom tokenizers we need to train them.

### Train your own tokenizer 

This would be an example training a tokenizer on Romeo and Juliet text.

```python
!wget https://raw.githubusercontent.com/dejanbatanjac/pytorch-learning-101/master/rij.txt
pip install tokenizers
from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers, processors

# initialize as simple BPE
tokenizer = Tokenizer(models.BPE())

# customize to use prefix and byte level decoding
tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)
tokenizer.decoder = decoders.ByteLevel()
tokenizer.post_processor = processors.ByteLevel(trim_offsets=True)

# train on a rij text
trainer = trainers.BpeTrainer(vocab_size=20000, min_frequency=3)
tokenizer.train(trainer, ["rij.txt"])

# encode a text
encoded = tokenizer.encode("I can feel the magic, can you?")
print(encoded.tokens)
# ['ĠI', 'Ġcan', 'Ġfeel', 'Ġthe', 'Ġm', 'ag', 'ic', ',', 'Ġcan', 'Ġyou', '?']
print(encoded.ids)
[87, 324, 1272, 80, 74, 733, 357, 3, 324, 116, 8]

d = tokenizer.get_vocab()
len(d) # 2608
```

We used the novel corpus and used byte level of BPE (ByteLevelBPETokenizer).
To save the tokenizer you may use:

```python
help(tokenizer.model)
```

The **tokenizer.train** can take multiple documents to came up to a vocab. We use 
**tokenizer.encode** method to encode a string before feeding the Transformer models.

If this is not obvious I just wanted to add that we have two trainings:
* tokenizer training (training the tokenizer)
* transformer model training (training the model)

## Save the Tokenizer

Once we have the tokenizers for the specific text (language, language model) we can save it. It is basically two files:

* vocab.json
* merges.txt

```python
!mkdir RIJTokenizer
tokenizer.save("RIJTokenizer")
!zip -r tok.zip RIJTokenizerif 
# download tok.zip
```

## Models

```python
# get random seed
seed = 12
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
```


## Save Model

trainer.save_model("./EsperBERTo")





<!-- 
## Appendix

### Tokenization details

For more details on word tokenization check out <iframe width="560" height="315" src="https://www.youtube.com/embed/XkImc447pZo" frameborder="0" allow="accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

### History of tokenization

For the history review of tokenization I am assuming this may be the list:
> * split function (by space, tab)
> * regular expression split
> * NLTK
> * Spacy
> * Gensim -->
