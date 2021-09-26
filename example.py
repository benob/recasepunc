import sys
from recasepunc import CasePuncPredictor

predictor = CasePuncPredictor('fr-txt.large.19000')

text = ' '.join(sys.argv[1:])
tokens = list(enumerate(predictor.tokenize(text)))

for token, case_label, punc_label in predictor.predict(tokens, lambda x: x[1]):
    print(token, case_label, punc_label, predictor.map_punc_label(predictor.map_case_label(token[1], case_label), punc_label))
