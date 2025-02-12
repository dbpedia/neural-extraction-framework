"""Functions related to BERT or similar models"""

from typing import List, Tuple

import numpy as np  # type: ignore
from transformers import AutoModel, AutoTokenizer  # type: ignore

from coref.config import Config
from coref.const import Doc
import datetime, pytz, time

IST = pytz.timezone("Asia/Kolkata")  # your time zone here


def get_subwords_batches(doc: Doc, config: Config, tok: AutoTokenizer) -> np.ndarray:
    """
    Turns a list of subwords to a list of lists of subword indices
    of max length == batch_size (or shorter, as batch boundaries
    should match sentence boundaries). Each batch is enclosed in cls and sep
    special tokens.

    Returns:
        batches of bert tokens [n_batches, batch_size]
    """
    # print('bert_window_size',config.bert_window_size, end='\r')
    # time.sleep(2)
    # print('O',time.time(),end=' ',flush=True)
    # print('=',doc['document_id'],'-'*10,end='\r')
    # time.sleep(5)
    # print(doc.keys())
    # input('wait')
    batch_size = config.bert_window_size - 2  # to save space for CLS and SEP

    subwords: List[str] = doc["subwords"]
    subwords_batches = []
    start, end = 0, 0

    while end < len(subwords):
        end = min(end + batch_size, len(subwords))

        # Move back till we hit a sentence end
        if end < len(subwords):
            sent_id = doc["sent_id"][doc["word_id"][end]]
            while end and doc["sent_id"][doc["word_id"][end - 1]] == sent_id:
                end -= 1

        # ritwik added this
        if (
            start >= end
        ):  # this means that the sentence on which start is standing is so long that it exceeds the batch_size
            end = min(start + batch_size, len(subwords))

        length = end - start
        batch = [tok.cls_token] + subwords[start:end] + [tok.sep_token]
        batch_ids = [-1] + list(range(start, end)) + [-1]

        # Padding to desired length
        # -1 means the token is a special token
        batch += [tok.pad_token] * (batch_size - length)
        batch_ids += [-1] * (batch_size - length)

        # if doc['document_id'] == 'bn/cts/02/cts_0213':
        #     print('\nbatch',batch,end='\n')
        #     print(subwords)
        #     print('batch_size',batch_size)
        #     print(start, end, len(subwords))
        #     input('wait')
        subwords_batches.append([tok.convert_tokens_to_ids(token) for token in batch])
        # if doc['document_id'] == 'bn/cts/02/cts_0213':
        #     print('done')
        start += length

    # print('C',time.time(),'-'*50,end='\r',flush=True)
    return np.array(subwords_batches)


def load_bert(config: Config) -> Tuple[AutoModel, AutoTokenizer]:
    """
    Loads bert and bert tokenizer as pytorch modules.

    Bert model is loaded to the device specified in config.device
    """
    print(f"Loading {config.bert_model}...")
    # print(config.device)
    # input('press enter')

    base_bert_name = config.bert_model.split("/")[-1]
    tokenizer_kwargs = config.tokenizer_kwargs.get(base_bert_name, {})
    if tokenizer_kwargs:
        print(f"Using tokenizer kwargs: {tokenizer_kwargs}")
    tokenizer = AutoTokenizer.from_pretrained(config.bert_model, **tokenizer_kwargs)

    model = AutoModel.from_pretrained(config.bert_model).to(config.device)

    print("Bert successfully loaded.")

    return model, tokenizer
