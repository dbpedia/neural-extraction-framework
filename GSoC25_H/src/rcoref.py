import jsonlines
import torch, time
from tqdm import tqdm
from coref import CorefModel
from coref.tokenizer_customization import *
from coref.mcoref import resolve_pronouns_hindi
import stanza
import re

if __name__ == "__main__":
    nlp = stanza.Pipeline("hi")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    raw_sentences = [
        "हमने कोहरे में एक आदमी को भागते देखा।  उसने काले रंग की शाल पहनी थी।  उसका कद लम्बा था। ",
        "मार्च 2001 में एक बार फिर अमरीका के अट्ठाइसवें रक्षा सचिव के रूप में उनकी वापसी हुई।",
    ]
    docs, out_sentences = resolve_pronouns_hindi(raw_sentences, model, nlp)
    out_sentences = out_sentences[0].split(".")
    assert len(raw_sentences) == len(out_sentences)
    for x, y in zip(raw_sentences, out_sentences):
        print("In ", x)
        print("Out", y, end="\n\n")
