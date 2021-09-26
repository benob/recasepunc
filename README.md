Recasing and punctuation model based on Bert
============================================
Benoit Favre 2021


This system converts a sequence of lowercase tokens without punctuation to a sequence of cased tokens with punctuation.

It is trained to predict both aspects at the token level in a multitask fashion, from fine-tuned BERT representations.

The model predicts the following recasing labels:
- lower: keep lowercase
- upper: convert to upper case
- capitalize: set first letter as upper case
- other: left as is

And the following punctuation labels:
- o: no punctuation
- period: .
- comma: ,
- question: ?
- exclamation: !

Input tokens are batched as sequences of length 256 that are processed independently without overlap.

In training, batches containing less that 256 tokens are simulated by drawing
uniformly a length and replacing all tokens and labels after that point with
padding (called Cut-drop).

Changelong:
* Fix generation when input is smaller than max length

Installation
------------

Use your favourite method for installing Python requirements. For example:
```
python -mvenv env
. env/bin/activate
pip3 install -r requirements.txt -f https://download.pytorch.org/whl/torch_stable.html
```


Prediction
----------

Predict from raw text:
```
python recasepunc.py predict checkpoint/path.iteration < input.txt > output.txt
```


Models
------

* French: [fr-txt.large.19000](https://talep-archives.lis-lab.fr/transfer/models/recasepunc/fr-txt.large.19000) trained on 160M tokens from Common Crawl
  * Iterations: 19000
  * Batch size: 16
  * Max length: 256
  * Seed: 871253
  * Cut-drop probability: 0.1
  * Train loss: 0.021128975618630648
  * Valid loss: 0.015684964135289192
  * Recasing accuracy: 96.73
  * Punctuation accuracy: 95.02
    * All punctuation F-score: 67.79
    * Comma F-score: 67.94
    * Period F-score: 72.91 
    * Question F-score: 57.57
    * Exclamation mark F-score: 15.78
  * Training data: First 100M words from [Common Crawl](http://data.statmt.org/cc-100/fr.txt.xz])


Training 
--------

Notes: You need to modify file names adequately.  Training tensors are precomputed and loaded in CPU memory.

Stage 0: download text data

Stage 1: tokenize and normalize text with Moses tokenizer, and extract recasing and repunctuation labels
```
python recasepunc.py preprocess < input.txt > input.case+punc
```

Stage 2: sub-tokenize with Flaubert tokenizer, and generate pytorch tensors
```
python recasepunc.py tensorize input.case+punc input.case+punc.x input.case+punc.y
```

Stage 3: train model
```
python recasepunc.py train train.x train.y valid.x valid.y checkpoint/path
```

Stage 4: evaluate performance on a test set 
```
python recasepunc.py eval checkpoint/path.iteration test.x test.y
```

