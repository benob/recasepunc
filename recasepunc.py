import sys
import collections
import os
import regex as re
from mosestokenizer import *
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import numpy as np
from torch.utils.data import TensorDataset, DataLoader

from transformers import AutoModel, AutoTokenizer

seed = 871253
lang = 'fr'
flavor = 'flaubert/flaubert_base_uncased'
max_length = 256
batch_size = 16
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# make sure everything is deterministic
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
torch.use_deterministic_algorithms(True)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
random.seed(seed)
np.random.seed(seed)

# NOTE: it is assumed in the implementation that y[:,0] is the punctuation label, and y[:,1] is the case label!

punctuation = {
    'O': 0,
    'COMMA': 1,
    'PERIOD': 2,
    'QUESTION': 3,
    'EXCLAMATION': 4,
}

punctuation_syms = ['', ',', '.', ' ?', ' !']

case = {
    'LOWER': 0,
    'UPPER': 1,
    'CAPITALIZE': 2,
    'OTHER': 3,
}

tokenizer = AutoTokenizer.from_pretrained(flavor, do_lower_case=True) 
pad_token_id = tokenizer.pad_token_id
cls_token_id = tokenizer.bos_token_id
cls_token = tokenizer.bos_token
sep_token_id = tokenizer.sep_token_id
sep_token = tokenizer.sep_token

class Model(nn.Module):
    def __init__(self, flavor=flavor, device=device):
        super().__init__()
        self.bert = AutoModel.from_pretrained(flavor)
        self.punc = nn.Linear(self.bert.dim, 5)
        self.case = nn.Linear(self.bert.dim, 4)
        self.dropout = nn.Dropout(0.3)
        self.to(device)

    def forward(self, x):
        output = self.bert(x)
        representations = self.dropout(F.gelu(output['previous_hidden_state']))
        punc = self.punc(representations)
        case = self.case(representations)
        return punc, case


# randomly create sequences that align to punctuation boundaries
def drop_at_boundaries(rate, x, y):
    #num_dropped = 0
    for i, dropped in enumerate(torch.rand((len(x),)) < rate):
        if dropped:
            # select all indices that are sentence endings
            indices = (y[i,:,0] > 1).nonzero(as_tuple=True)[0]
            if len(indices) < 2:
                continue
            start = indices[0] + 1
            end = indices[random.randint(1, len(indices) - 1)] + 1
            length = end - start
            if length + 2 > len(x[i]):
                continue
            #print(y[i,:,0])
            #print(x[i].tolist())
            #print(y[i,:,0].tolist())
            #print(y[i,:,1].tolist())
            #print(indices.tolist())
            #print(start.item(), end.item(), length.item())
            x[i, 0] = cls_token_id
            x[i, 1: length + 1] = x[i, start: end].clone()
            x[i, length + 1] = sep_token_id
            x[i, length + 2:] = pad_token_id
            y[i, 0] = 0
            y[i, 1: length + 1] = y[i, start: end].clone()
            y[i, length + 1:] = 0
            #print(x[i].tolist())
            #print(y[i,:,0].tolist())
            #print(y[i,:,1].tolist())
            #print()
            #num_dropped += 1
    #print(num_dropped / len(x))


def compute_performance(model, loader):
    criterion = nn.CrossEntropyLoss()
    model.eval()
    total_loss = all_correct1 = all_correct2 = num_loss = num_perf = 0
    num_ref = collections.defaultdict(float)
    num_hyp = collections.defaultdict(float)
    num_correct = collections.defaultdict(float)
    for x, y in loader:
        x = x.long().to(device)
        y = y.long().to(device)
        y1 = y[:,:,0]
        y2 = y[:,:,1]
        with torch.no_grad():
            y_scores1, y_scores2 = model(x.to(device))
            loss1 = criterion(y_scores1.view(y1.size(0) * y1.size(1), -1), y1.view(y1.size(0) * y1.size(1)))
            loss2 = criterion(y_scores2.view(y2.size(0) * y2.size(1), -1), y2.view(y2.size(0) * y2.size(1)))
            loss = loss1 + loss2
            y_pred1 = torch.max(y_scores1, 2)[1]
            y_pred2 = torch.max(y_scores2, 2)[1]
            for label in range(1, 6):
                ref = (y1 == label)
                hyp = (y_pred1 == label)
                correct = (ref * hyp == 1)
                num_ref[label] += ref.sum()
                num_hyp[label] += hyp.sum()
                num_correct[label] += correct.sum()
                num_ref[0] += ref.sum()
                num_hyp[0] += hyp.sum()
                num_correct[0] += correct.sum()
            all_correct1 += (y_pred1 == y1).sum()
            all_correct2 += (y_pred2 == y2).sum()
            total_loss += loss.item()
            num_loss += len(y)
            num_perf += len(y) * max_length 
    recall = {}
    precision = {}
    fscore = {}
    for label in range(0, 6):
        recall[label] = num_correct[label] / num_ref[label] if num_ref[label] > 0 else 0
        precision[label] = num_correct[label] / num_hyp[label] if num_hyp[label] > 0 else 0
        fscore[label] = (2 * recall[label] * precision[label] / (recall[label] + precision[label])).item() if recall[label] + precision[label] > 0 else 0
    return total_loss / num_loss, all_correct2.item() / num_perf, all_correct1.item() / num_perf, fscore


def fit(model, checkpoint_path, train_loader, valid_loader, iterations, valid_period=200, lr=1e-5):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda param: param.requires_grad, model.parameters()), lr=lr)
    iteration = 0
    while True:
        model.train()
        total_loss = num = 0
        for x, y in tqdm(train_loader):
            x = x.long().to(device)
            y = y.long().to(device)
            drop_at_boundaries(0.1, x, y)
            y1 = y[:,:,0]
            y2 = y[:,:,1]
            optimizer.zero_grad()
            y_scores1, y_scores2 = model(x)
            loss1 = criterion(y_scores1.view(y1.size(0) * y1.size(1), -1), y1.view(y1.size(0) * y1.size(1)))
            loss2 = criterion(y_scores2.view(y2.size(0) * y2.size(1), -1), y2.view(y2.size(0) * y2.size(1)))
            loss = loss1 + loss2
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            num += len(y)
            if iteration % valid_period == valid_period - 1:
                train_loss = total_loss / num
                valid_loss, valid_accuracy_case, valid_accuracy_punc, valid_fscore = compute_performance(model, valid_loader)
                torch.save({
                    'iteration': iteration + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_loss': train_loss,
                    'valid_loss': valid_loss,
                    'valid_accuracy_case': valid_accuracy_case,
                    'valid_accuracy_punc': valid_accuracy_punc,
                    'valid_fscore': valid_fscore,
                }, '%s.%d' % (checkpoint_path, iteration + 1))
                print(iteration + 1, train_loss, valid_loss, valid_accuracy_case, valid_accuracy_punc, valid_fscore)
                total_loss = num = 0

            iteration += 1
            if iteration > iterations:
                return

            sys.stdout.flush()
            sys.stdout.flush()


def batchify(x, y):
    x = x[:(len(x) // max_length) * max_length].reshape(-1, max_length)
    y = y[:(len(y) // max_length) * max_length, :].reshape(-1, max_length, 2)
    return x, y


def train(train_x_fn, train_y_fn, valid_x_fn, valid_y_fn, checkpoint_path):
    X_train, Y_train = batchify(torch.load(train_x_fn), torch.load(train_y_fn))
    X_valid, Y_valid = batchify(torch.load(valid_x_fn), torch.load(valid_y_fn))

    train_set = TensorDataset(X_train, Y_train)
    valid_set = TensorDataset(X_valid, Y_valid)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_set, batch_size=batch_size)

    model = Model()

    fit(model, checkpoint_path, train_loader, valid_loader, 20000, 1000)


def run_eval(test_x_fn, test_y_fn, checkpoint_path):
    X_test, Y_test = batchify(torch.load(test_x_fn), torch.load(test_y_fn))
    test_set = TensorDataset(X_test, Y_test)
    test_loader = DataLoader(test_set, batch_size=batch_size)

    model = Model()
    loaded = torch.load(checkpoint_path)
    model.load_state_dict(loaded['model_state_dict'])

    print(*compute_performance(model, test_loader))


def recase(token, label):
    if label == case['LOWER']:
        return token.lower()
    elif label == case['CAPITALIZE']:
        return token.lower().capitalize()
    elif label == case['UPPER']:
        return token.upper()
    else:
        return token


class CasePuncPredictor:
    def __init__(self, checkpoint_path, flavor=flavor, device=device):
        self.model = Model(flavor)
        loaded = torch.load(checkpoint_path, map_location=device)
        self.model.load_state_dict(loaded['model_state_dict'])

        self.tokenizer = AutoTokenizer.from_pretrained(flavor, do_lower_case=True) 

        self.rev_case = {b: a for a, b in case.items()}
        self.rev_punc = {b: a for a, b in punctuation.items()}

    def tokenize(self, text):
        return [cls_token] + tokenizer.tokenize(text.lower()) + [sep_token]

    def predict(self, tokens, getter=lambda x: x, max_length=max_length, device=device):
        if type(tokens) == str:
            tokens = self.tokenize(tokens)
        previous_label = punctuation['PERIOD']
        for start in range(0, len(tokens), max_length):
            instance = tokens[start: start + max_length]
            if type(getter(instance[0])) == str:
                ids = self.tokenizer.convert_tokens_to_ids(getter(token) for token in instance)
            else:
                ids = [getter(token) for token in instance]
            if len(ids) < max_length:
                ids += [0] * (max_length - len(ids))
            x = torch.tensor([ids]).long().to(device)
            y_scores1, y_scores2 = self.model(x)
            y_pred1 = torch.max(y_scores1, 2)[1]
            y_pred2 = torch.max(y_scores2, 2)[1]
            for i, id, token, punc_label, case_label in zip(range(len(instance)), ids, instance, y_pred1[0].tolist()[:len(instance)], y_pred2[0].tolist()[:len(instance)]):
                if id == cls_token_id or id == sep_token_id:
                    continue
                if previous_label != None and previous_label > 1:
                    if case_label in [0, 3]: # LOWER, OTHER
                        case_label = case['CAPITALIZE']
                if i + start == len(tokens) - 2 and punc_label == punctuation['O']:
                    punc_label = punctuation['PERIOD']
                yield (token, self.rev_case[case_label], self.rev_punc[punc_label])
                previous_label = punc_label

    def map_case_label(self, token, case_label):
        if token.endswith('</w>'):
            token = token[:-4]
        return recase(token, case[case_label])

    def map_punc_label(self, token, punc_label):
        if token.endswith('</w>'):
            token = token[:-4]
        return token + punctuation_syms[punctuation[punc_label]]


def generate_predictions(checkpoint_path, debug=False):
    model = Model()
    loaded = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(loaded['model_state_dict'])

    tokenizer = AutoTokenizer.from_pretrained(flavor, do_lower_case=True) 

    rev_case = {b: a for a, b in case.items()}
    rev_punc = {b: a for a, b in punctuation.items()}

    for line in sys.stdin:
        tokens = [cls_token] + tokenizer.tokenize(line.lower()) + [sep_token]
        was_word = False
        previous_label = punctuation['PERIOD']
        for start in range(0, len(tokens), max_length):
            instance = tokens[start: start + max_length]
            ids = tokenizer.convert_tokens_to_ids(instance)
            #print(len(ids), file=sys.stderr)
            if len(ids) < max_length:
                ids += [0] * (max_length - len(ids))
            x = torch.tensor([ids]).long().to(device)
            y_scores1, y_scores2 = model(x)
            y_pred1 = torch.max(y_scores1, 2)[1]
            y_pred2 = torch.max(y_scores2, 2)[1]
            for id, token, punc_label, case_label in zip(ids, instance, y_pred1[0].tolist()[:len(instance)], y_pred2[0].tolist()[:len(instance)]):
                if debug:
                    print(id, token, punc_label, case_label, file=sys.stderr)
                if id == cls_token_id or id == sep_token_id:
                    continue
                if previous_label != None and previous_label > 1:
                    if case_label in [0, 3]: # LOWER, OTHER
                        case_label = case['CAPITALIZE']
                previous_label = punc_label
                if token.endswith('</w>'):
                    cased_token = recase(token[:-4], case_label)
                    if was_word:
                        print(' ', end='')
                    print(cased_token + punctuation_syms[punc_label], end='')
                    was_word = True
                else:
                    cased_token = recase(token, case_label)
                    if was_word:
                        print(' ', end='')
                    print(cased_token, end='')
                    was_word = False
        if previous_label == 0:
            print('.', end='')
        print()


def label_for_case(token):
    token = re.sub('[^\p{Ll}\p{Lu}]', '', token)
    if token == token.lower():
        return 'LOWER'
    elif token == token.lower().capitalize():
        return 'CAPITALIZE'
    elif token == token.upper():
        return 'UPPER'
    else:
        return 'OTHER'


def make_tensors(input_fn, output_x_fn, output_y_fn, debug=False):
    tokenizer = AutoTokenizer.from_pretrained(flavor, do_lower_case=True) 
    with open(input_fn) as fp:
        lines = fp.readlines()
        size = int(1.5 * len(lines))
        X = torch.IntTensor(size)
        Y = torch.ByteTensor(size, 2)

        offset = 0
        for line in lines:
            word, punc_label = line.strip().split('\t')
            case_label = label_for_case(word)
            tokens = tokenizer.tokenize(word.lower())
            ids = tokenizer.convert_tokens_to_ids(tokens)
            for i, (id, token) in enumerate(zip(ids, tokens)):
                if i > 0 and case_label == 'CAPITALIZE':
                    case_label = 'LOWER'
                X[offset] = id
                Y[offset, 0] = punctuation[punc_label if i == len(ids) - 1 else 'O']
                Y[offset, 1] = case[case_label]
                if debug:
                    print(word, token, id, punc_label if i == len(ids) - 1 else 'O', case_label, file=sys.stderr)
                offset += 1

        print(size, offset)
        torch.save(X[:offset], output_x_fn)
        torch.save(Y[:offset], output_y_fn)


def preprocess_text():
    mapped_punctuation = {
        '.': 'PERIOD',
        '...': 'PERIOD',
        ',': 'COMMA',
        ';': 'COMMA',
        ':': 'COMMA',
        '(': 'COMMA',
        ')': 'COMMA',
        '?': 'QUESTION',
        '!': 'EXCLAMATION',
    }
    splitsents = MosesSentenceSplitter(lang)
    tokenize = MosesTokenizer(lang, extra=['-no-escape'])
    normalize = MosesPunctuationNormalizer(lang)

    for line in sys.stdin:
        if line.strip() != '':
            for sentence in splitsents([normalize(line)]):
                tokens = tokenize(sentence)
                previous_token = None
                for token in tokens:
                    if token in punctuation:
                        if previous_token != None:
                            print(previous_token, punctuation[token], sep='\t')
                        previous_token = None
                    elif not re.search('[\p{Ll}\p{Lu}\d]', token): # remove non-alphanumeric tokens
                        continue
                    else:
                        if previous_token != None:
                            print(previous_token, 'O', sep='\t')
                        previous_token = token
                if previous_token != None:
                    print(previous_token, 'PERIOD', sep='\t')
            
if __name__ == '__main__':
    command = sys.argv[1]
    if command == 'train':
        train(*sys.argv[2:])
    elif command == 'eval':
        run_eval(*sys.argv[2:])
    elif command == 'predict': 
        generate_predictions(*sys.argv[2:])
    elif command == 'tensorize':
        make_tensors(*sys.argv[2:])
    elif command == 'preprocess':
        preprocess_text()
    else:
        print('usage: %s train|eval|predict|tensorize|preprocess' % sys.argv[0]) 
        sys.exit(1) 
