import sys
from recasepunc import CasePuncPredictor, \
    WordpieceTokenizer  # WordpieceTokenizer is necessary for flavors other than fr

predictor = CasePuncPredictor('it.22000')

text = ' '.join(sys.argv[1:])
tokens = list(enumerate(predictor.tokenize(text)))

for token, case_label, punc_label in predictor.predict(tokens, lambda x: x[1]):
    print(
        token, case_label, punc_label, predictor.map_punc_label(
            predictor.map_case_label(token[1], case_label), punc_label
        )
    )
    # print(predictor.map_punc_label(predictor.map_case_label(token[1], case_label), punc_label))
