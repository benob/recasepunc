Recasing and punctuation model based on Bert
============================================
Benoit Favre 2021


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

[checkpoints/it.22000](https://github.com/CoffeePerry/recasepunc/releases/download/v0.1.0/it.22000)
```
{
  "iteration": "22000",
  "train_loss": "0.058934884114190934",
  "valid_loss": "0.06988634882792658",
  "valid_accuracy_case": "0.9575860089785607",
  "valid_accuracy_punc": "0.940614491584733",
  "valid_fscore": "{0: 0.6431694030761719, 1: 0.6150795817375183, 2: 0.7023577094078064, 3: 0.5514711737632751, 4: 0.21250930428504944}",
  "config": "{'seed': 871253, 'lang': 'it', 'flavor': 'dbmdz/bert-base-italian-uncased', 'max_length': 256, 'batch_size': 4, 'updates': 24000, 'period': 1000, 'lr': 1e-05, 'dab_rate': 0.1, 'device': device(type='cuda'), 'debug': False, 'action': 'train', 'action_args': ['data/it-100M.train.x', 'data/it-100M.train.y', 'data/it-100M.valid.x', 'data/it-100M.valid.y', 'checkpoints/it'], 'pad_token_id': 0, 'cls_token_id': 102, 'cls_token': '[CLS]', 'sep_token_id': 103, 'sep_token': '[SEP]'}"
}
```

[checkpoints/zh.24000](https://github.com/benob/recasepunc/releases/download/0.3/zh.24000)
```
{
  "iteration": "24000",
  "train_loss": "0.006788245493080467",
  "valid_loss": "0.007345725328494341",
  "valid_accuracy_case": "0.9963942307692307",
  "valid_accuracy_punc": "0.9692508012820513",
  "valid_fscore": "{0: 0.7727023363113403, 1: 0.7901785373687744, 2: 0.7293065190315247, 3: 0.7692307829856873, 4: 0.4615384638309479}",
  "config": "{'seed': 871253, 'lang': 'zh', 'flavor': 'ckiplab/bert-base-chinese', 'max_length': 256, 'batch_size': 16, 'updates': 24000, 'period': 1000, 'lr': 1e-05, 'dab_rate': 0.1, 'device': device(type='cuda'), 'debug': False, 'action': 'train', 'action_args': ['data/zh-100M.train.x', 'data/zh-100M.train.y', 'data/zh-100M.valid.x', 'data/zh-100M.valid.y', 'checkpoints/zh'], 'pad_token_id': 0, 'cls_token_id': 101, 'cls_token': '[CLS]', 'sep_token_id': 102, 'sep_token': '[SEP]'}"
}
```

[checkpoints/en.23000](https://github.com/benob/recasepunc/releases/download/0.3/en.23000)
```
{
  "iteration": "23000",
  "train_loss": "0.014598741472698748",
  "valid_loss": "0.025432642453756087",
  "valid_accuracy_case": "0.9407051282051282",
  "valid_accuracy_punc": "0.9401041666666666",
  "valid_fscore": "{0: 0.6455026268959045, 1: 0.5925925970077515, 2: 0.7243649959564209, 3: 0.7027027010917664, 4: 0.03921568766236305}",                                                    
  "config": "{'seed': 871253, 'lang': 'en', 'flavor': 'bert-base-uncased', 'max_length': 256, 'batch_size': 16, 'updates': 24000, 'period': 1000, 'lr': 1e-05, 'dab_rate': 0.1, 'device': device(type='cuda'), 'debug': False, 'action': 'train', 'action_args': ['data/en-100M.train.x', 'data/en-100M.train.y', 'data/en-100M.valid.x', 'data/en-100M.valid.y', 'checkpoints/en'], 'pad_token_id': 0, 'cls_token_id': 101, 'cls_token': '[CLS]', 'sep_token_id': 102, 'sep_token': '[SEP]'}"                                                                                           
}
```

[checkpoints/fr.22000](https://github.com/benob/recasepunc/releases/download/0.3/fr.22000)
```
{
  "iteration": "22000",
  "train_loss": "0.02052250287961215",
  "valid_loss": "0.009240646392871171",
  "valid_accuracy_case": "0.9881810897435898",
  "valid_accuracy_punc": "0.9683493589743589",
  "valid_fscore": "{0: 0.802524745464325, 1: 0.7892595529556274, 2: 0.8360477685928345, 3: 0.8717948198318481, 4: 0.2068965584039688}",
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

Stage 2: sub-tokenize with Flaubert tokenizer, and generate pytorch tensors
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

Notes
-----

This work was not published, but a similar model is described in "FullStop: Multilingual Deep Models for Punctuation Prediction", Frank et al, 2021.
