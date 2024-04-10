Recasing and punctuation model based on Bert
============================================
Benoit Favre 2021-2024


This system converts a sequence of lowercase tokens without punctuation to a sequence of cased tokens with punctuation.

It is trained to predict both aspects at the token level in a multitask fashion, from fine-tuned BERT representations.

The model predicts the following recasing labels:
- lower: keep lowercase
- upper: convert to upper case
- capitalize: set first letter as upper case
- other: left as is, but could be processed with a list

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

Changelog:
* v0.4: Retrain with latest transformers
* Add support for Zh and En models
* Fix generation when input is smaller than max length

Installation
------------

Use your favourite method for installing Python requirements. For example:
```
python -m venv env
. env/bin/activate
pip3 install -r requirements.txt -f https://download.pytorch.org/whl/torch_stable.html
```


Prediction
----------

Predict from raw text:
```
python recasepunc.py predict checkpoint/path < input.txt > output.txt
```


Models
------

All models are trained from the 1st 100M tokens from [Common Crawl](http://data.statmt.org/cc-100/)

[checkpoints/it.23000](https://github.com/benob/recasepunc/releases/download/v0.4/it.23000)
```
{
  "iteration": "23000",
  "train_loss": "0.015077149430289864",
  "valid_loss": "0.021484553813934326",
  "valid_accuracy_case": "0.9517227564102564",
  "valid_accuracy_punc": "0.9359975961538461",
  "valid_fscore": "{0: 0.6016615629196167, 1: 0.6202345490455627, 2: 0.6219512224197388, 3: 0.42424243688583374, 4: 0.08571428805589676}",
  "config": "{'seed': 871253, 'lang': 'it', 'flavor': 'dbmdz/bert-base-italian-uncased', 'max_length': 256, 'batch_size': 16, 'updates': 24000, 'period': 1000, 'lr': 1e-05, 'dab_rate': 0.1, 'device': device(type='cuda'), 'debug': False, 'action': 'train', 'action_args': ['data/it-100M.train.x', 'data/it-100M.train.y', 'data/it-100M.valid.x', 'data/it-100M.valid.y', 'checkpoints/it'], 'pad_token_id': 0, 'cls_token_id': 102, 'cls_token': '[CLS]', 'sep_token_id': 103, 'sep_token': '[SEP]'}"
}
```

[checkpoints/zh-Hant.17000](https://github.com/benob/recasepunc/releases/download/0.4/zh-Hant.17000)
```
{
  "iteration": "17000",
  "train_loss": "0.007012549160048366",
  "valid_loss": "0.007463883130978315",
  "valid_accuracy_case": "0.9967948717948718",
  "valid_accuracy_punc": "0.9682491987179487",
  "valid_fscore": "{0: 0.7668336033821106, 1: 0.7813194990158081, 2: 0.7200000286102295, 3: 0.8333333730697632, 4: 0.7272727489471436}",
  "config": "{'seed': 871253, 'lang': 'zh-Hant', 'flavor': 'ckiplab/bert-base-chinese', 'max_length': 256, 'batch_size': 16, 'updates': 24000, 'period': 1000, 'lr': 1e-05, 'dab_rate': 0.1, 'device': device(type='cuda'), 'debug': False, 'action': 'train', 'action_args': ['data/zh-Hant-100M.train.x', 'data/zh-Hant-100M.train.y', 'data/zh-Hant-100M.valid.x', 'data/zh-Hant-100M.valid.y', 'checkpoints/zh-Hant'], 'pad_token_id': 0, 'cls_token_id': 101, 'cls_token': '[CLS]', 'sep_token_id': 102, 'sep_token': '[SEP]'}"
}
```

[checkpoints/en.22000](https://github.com/benob/recasepunc/releases/download/0.4/en.22000)
```
{
  "iteration": "22000",
  "train_loss": "0.01467611983884126",
  "valid_loss": "0.02559371789296468",
  "valid_accuracy_case": "0.9393028846153846",
  "valid_accuracy_punc": "0.9404046474358975",
  "valid_fscore": "{0: 0.6431096196174622, 1: 0.603951096534729, 2: 0.7078340649604797, 3: 0.6865671277046204, 4: 0}",
  "config": "{'seed': 871253, 'lang': 'en', 'flavor': 'bert-base-uncased', 'max_length': 256, 'batch_size': 16, 'updates': 24000, 'period': 1000, 'lr': 1e-05, 'dab_rate': 0.1, 'device': device(type='cuda'), 'debug': False, 'action': 'train', 'action_args': ['data/en-100M.train.x', 'data/en-100M.train.y', 'data/en-100M.valid.x', 'data/en-100M.valid.y', 'checkpoints/en'], 'pad_token_id': 0, 'cls_token_id': 101, 'cls_token': '[CLS]', 'sep_token_id': 102, 'sep_token': '[SEP]'}"
}
```

[checkpoints/fr.24000](https://github.com/benob/recasepunc/releases/download/0.4/fr.24000)
```
{
  "iteration": "24000",
  "train_loss": "0.015482447233051061",
  "valid_loss": "0.006200919071069131",
  "valid_accuracy_case": "1.0",
  "valid_accuracy_punc": "0.9691506410256411",
  "valid_fscore": "{0: 0.8114132881164551, 1: 0.7968379855155945, 2: 0.8446389436721802, 3: 0.8421052694320679, 4: 0.3076923191547394}",
  "config": "{'seed': 871253, 'lang': 'fr', 'flavor': 'flaubert/flaubert_base_uncased', 'max_length': 256, 'batch_size': 16, 'updates': 24000, 'period': 1000, 'lr': 1e-05, 'dab_rate': 0.1, 'device': device(type='cuda'), 'debug': False, 'action': 'train', 'action_args': ['data/fr-100M.train.x', 'data/fr-100M.train.y', 'data/fr-100M.valid.x', 'data/fr-100M.valid.y', 'checkpoints/fr'], 'pad_token_id': 2, 'cls_token_id': 0, 'cls_token': '<s>', 'sep_token_id': 1, 'sep_token': '</s>'}"
}
```


Training 
--------

Notes: You need to modify file names adequately.  Training tensors are precomputed and loaded to CPU memory, models and batches are moved to CUDA memory.

Stage 0: download text data

Stage 1: tokenize and normalize text with Moses tokenizer, and extract recasing and repunctuation labels
```
python recasepunc.py preprocess --lang $LANG < input.txt > input.case+punc
```

Stage 2: sub-tokenize with Flaubert/moses tokenizer, and generate pytorch tensors
```
python recasepunc.py tensorize input.case+punc input.case+punc.x input.case+punc.y --lang $LANG
```

Stage 3: train model
```
python recasepunc.py train train.x train.y valid.x valid.y checkpoint/path --lang $LANG
```

Stage 4: evaluate performance on a test set 
```
python recasepunc.py eval test.x test.y checkpoint/path.iteration
```

Two scripts used to create the models are given as example of how to train for a new language:
* `./prepare.sh <lang>` for downloading data, creating sets, and preprocessing
* `./train.sh <lang>` for trainging the model

Both assume the availability of a `env.sh` script for loading the environment and setting up stuff.
`requirements.freeze.txt` contains the package versions used for training.
You will need to modify recasepunc.py and set the BERT flavior for the new language and check that the tokenizer correctly handles punctuation. For French, we had to patch the tokenizer to keep input/punctuation synchronized.

Notes
-----

This work was not published, but a similar model is described in "FullStop: Multilingual Deep Models for Punctuation Prediction", Frank et al, 2021.
