import os, torch, random, pytz, glob, datetime, time, psutil, json, re
import pandas as pd, numpy as np
import matplotlib.pyplot as plt
from torch.utils import data
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
import torch.nn.functional as F
import torch.nn as nn
import string
from collections import Counter
from transformers import AdamW
from transformers.modeling_outputs import SequenceClassifierOutput
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)


def foreign_characters(s):
    # all unicode blocks here https://www.fileformat.info/info/unicode/block/index.htm
    ml = []
    s = re.sub(r"\(.*?\)", "", s)  # foreign characters can come inside round brackets
    s = re.sub(r"[A-Z]+", "", s)  # foreign characters can come as abbreviations
    s = re.sub(r"(mg)|(ml)|(khz)|(kg)", "", s.lower())  # removing common english terms

    # when i directly modify s, then it means that, these characters will NOT become a reason to drop the sentence in consideration
    s = re.sub(r"[\u0080-\u00FF]", "", s)  # Latin-1 Supplement: degree symbol
    s = re.sub(r"[\u02B0-\u02FF]", "", s)  # Spacing Modifier: ring above, dot above
    s = re.sub(
        r"[\u2000-\u209F]", "", s
    )  # General Punctuation: zero-width joiner + superscripts
    s = re.sub(r"[\u2200-\u22FF]", "", s)  # mathematical symbols
    s = re.sub(r"[\u0300-\u0362]", "", s)  # Diacritical Marks
    s = re.sub(
        r"[\u0020-\u0040]|[\u005B-\u0060]|[\u007B-\u007E]", "", s
    )  # Punctuations and Digit block: removing those which are common in all languages
    s = re.sub(r"[\u0964\u0965]", "", s)  # hindi single purna viram, double purna viram
    s = re.sub(r"[\uFE00-\uFE0F]+", "", s)  # emoji variations

    # when i make a new variable for s, then it means that, these characters WILL become a reason to drop the sentence in consideration
    # some erroneous languages, which were not supposed to be present but yet they are present
    s2 = re.sub(r"[\u0100-\u024F]", "", s)  # Latin extended - A and B
    s2 = re.sub(r"[\u1E00-\u1EFF]", "", s2)  # Latin Extended Additional
    s2 = re.sub(r"[\uFFF9-\uFFFF]", "", s2)  # special characters
    s2 = re.sub(
        r"[\u0300-\u05FF]", "", s2
    )  # some middle eastern languages, and russian
    s2 = re.sub(r"[\u20A0-\uA8DF]", "", s2)  # other obscure languages
    s2 = re.sub(r"[\uA900-\uFFFFF]", "", s2)  # other obscure languages
    s2 = re.sub(r"[\U0001F004-\U0001FA95]+", "", s2)  # emojis

    lang_unicodes = [
        ["English", ("\u0021", "\u007F")],
        ["Devnagri", ("\u0900", "\u097F"), ("\uA8E0", "\uA8FF")],
        ["Bangla", ("\u0980", "\u09FF")],
        ["Gujarati", ("\u0A80", "\u0AFF")],
        ["Urdu/Persian/Arabic", ("\u0600", "\u06FF"), ("\u08A0", "\u08FF")],
        ["Tamil", ("\u0B80", "\u0BFF")],
        ["Telegu", ("\u0C00", "\u0C7F")],
        ["punjabi/gurumukhi", ("\u0A00", "\u0A7F")],
        ["malayalam", ("\u0D00", "\u0D7F")],
        ["oriya", ("\u0B00", "\u0B7F")],
        ["kannada", ("\u0C80", "\u0CFF")],
        ["Sinhala", ("\u0D80", "\u0DFF")],
        ["Thai", ("\u0E00", "\u0E7F")],
        ["Lao", ("\u0E80", "\u0EFF")],
        ["Tibetan", ("\u0F00", "\u0FFF")],
        ["Greek", ("\u1F00", "\u1FFF")],
        # ,['',('\u','\u')], ['',('\u','\u')], ['',('\u','\u')], ['',('\u','\u')], ['',('\u','\u')], ['',('\u','\u')]
    ]

    for word in s2.split():
        for ch in word:
            found = False
            for i, lu in enumerate(lang_unicodes):
                for block in lu[1:]:
                    if ch >= block[0] and ch <= block[1]:
                        found = True
                        ml.append(i)
            if not found:
                pass
                # print('Wait!! Some unknown character encountered in character <'+ch+'> in word <'+word+'> in sentence <'+s2+'>')
                # print('Its unicode is',ch.encode('unicode_escape'))
                # input('Waiting...')
    if not ml:
        return True

    c = Counter(ml)
    base_script = c.most_common()[0][0]

    s = re.sub(r"\s", "", s)
    base_lang = lang_unicodes[base_script]
    foreign_character_found = True
    for block in base_lang[1:]:
        foreign_character_found = foreign_character_found and bool(
            re.search(r"[^" + block[0] + r"-" + block[1] + r"]+", s)
        )
        # if foreign_character_found:
        # 	 print('-'*100,re.search(r'[^'+block[0]+r'-'+block[1]+r']+',s)[0],re.search(r'[^'+block[0]+r'-'+block[1]+r']+',s)[0].encode('unicode_escape'))
        s = re.sub(r"[" + block[0] + r"-" + block[1] + r"]+", "", s)

    return foreign_character_found


class chunker_class(torch.nn.Module):
    def __init__(self, d, hyper_params):
        super(chunker_class, self).__init__()
        self.hyper_params = hyper_params
        self.model = AutoModel.from_pretrained(
            self.hyper_params["bert_model_name"],
            return_dict=True,
            output_hidden_states=True,
        )
        print("========== Model created ==========")
        self.fc1 = nn.Linear(
            hyper_params["embedding_size"], len(self.hyper_params["my_tagset"])
        )
        self.activation = nn.ReLU()
        self.criterion = torch.nn.CrossEntropyLoss()

        # self.optim = AdamW(self.model.parameters(), lr=alpha)
        self.device = d
        self.best_val_acc = -1

    def predict_from_logits(self, logits, attention_mask, error_fixing=False):
        if error_fixing:
            batch_tags = []
            last_B_tag_index, X_tag_index = 11, 24  # checked both of them manually
            for sent in logits:
                sent_tags, prev_tag = [], ""
                for word in sent:
                    word_tag, mx_idx = "", ""
                    mx_idx = word[: last_B_tag_index + 1].argmax()
                    if word[mx_idx] <= word[X_tag_index]:
                        mx_idx = X_tag_index
                    if prev_tag != "":
                        next_possible_tag = "I_" + prev_tag[2:]
                        if (
                            word[mx_idx]
                            <= word[
                                self.hyper_params["my_tagset"].index(next_possible_tag)
                            ]
                        ):
                            mx_idx = self.hyper_params["my_tagset"].index(
                                next_possible_tag
                            )
                    word_tag = self.hyper_params["my_tagset"][mx_idx]
                    prev_tag = (
                        word_tag if word_tag != "X" else prev_tag
                    )  # such that prev_tag is always = last non-X predicted tag
                    sent_tags.append(word_tag)
                batch_tags.append(sent_tags)
        else:
            a = torch.argmax(
                logits, dim=2
            )  # logits-->[batches,max_len,tag_len] now a-->[batches, max_len]
            batch_tags = []
            for b in a:
                t = [self.hyper_params["my_tagset"][c] for c in b]
                batch_tags.append(t)
        # batch_tags are like [ ['B_NP','I_NP',...,'X'], ['B_NP','B_NP',...,'X'] ]
        batch_tags_flat = [item for sublist in batch_tags for item in sublist]
        # batch_tags_flat are like ['B_NP','I_NP',...,'X','B_NP','B_NP',...,'X']
        batch_tags_pruned = [
            a2 for b, a2 in zip(attention_mask.view(-1) == 1, batch_tags_flat) if b
        ]
        # batch_tags_pruned are like ['B_NP','I_NP',...,'B_NP','B_NP',...]
        return batch_tags_pruned

    def take_mean(self, tensorlist):
        out = 0
        for t in tensorlist:
            out += t
        out /= len(tensorlist)
        return out

    def forward(
        self,
        input_ids,
        attention_mask,
        y,
        wordpiece_indices,
        save_embeddings=False,
        numpy_containers=[],
    ):
        if self.hyper_params["embedding_way"] == "last_hidden_state":
            out = self.model(
                input_ids, attention_mask=attention_mask
            ).last_hidden_state  # [batches, max_len, 768]
        elif self.hyper_params["embedding_way"] == "first_two":
            out = self.model(input_ids, attention_mask=attention_mask)
            out = self.take_mean(
                [out.hidden_states[0], out.hidden_states[1]]
            )  # [batches, max_len, 768]
        elif self.hyper_params["embedding_way"] == "last_two":
            out = self.model(input_ids, attention_mask=attention_mask)
            out = self.take_mean(
                [out.hidden_states[-2], out.hidden_states[-3]]
            )  # [batches, max_len, 768]
        else:
            raise "Unknown embedding_way specified"
        # out dimensions = [batches, max_len, 768]
        # y dimensions = [batches, max_len, len(tag_set)]
        # wordpiece_indices dimensions = [batches, max_len, 2]
        # print(out.shape, y.shape, wordpiece_indices.shape)
        # input('wait')

        if self.hyper_params["which_way"] == 3:
            c = torch.zeros(out.shape).to(self.device)
            for i in range(out.shape[0]):
                out_sent_i = out[i]
                wpi_sent_i = wordpiece_indices[i]
                wpi_sent_i = (
                    wpi_sent_i[wpi_sent_i != 0]
                ).tolist()  # this step flattens the list of pairs to a single list
                # print('$$',wpi_sent_i)
                j, k = 0, 0
                while j < out_sent_i.shape[0]:
                    if (
                        j not in wpi_sent_i[::2]
                    ):  # that is why we are hopping with 2 steps
                        c[i, k] += out[i, j]
                        j += 1
                    else:
                        start = j
                        end = wpi_sent_i[2 * wpi_sent_i[::2].index(j) + 1]
                        c[i, k] += torch.mean(out[i][start:end], 0)
                        j = end
                    k += 1
            out = c

        if save_embeddings and (
            numpy_containers[0].shape[0] < 5000 or numpy_containers[1].shape[0] < 5000
        ):
            subwords = numpy_containers[0]
            nonsubwords = numpy_containers[1]
            for batch_i in range(out.shape[0]):
                sent_i = out[batch_i]  # [max_len, 768]
                sent_i_b_nps = (y[batch_i][:, 5] == 1).nonzero(as_tuple=True)[
                    0
                ]  # indices of all B_NPs contained in a single vector 1xn
                wordpiece_absolute_indices = [wordpiece_indices[batch_i][0][0]]

                reduce_offset = (
                    wordpiece_indices[batch_i][0][1]
                    - wordpiece_indices[batch_i][0][0]
                    - 1
                )
                for i1 in range(1, wordpiece_indices.shape[1]):
                    if (
                        wordpiece_indices[batch_i][i1][0] == 0
                        and wordpiece_indices[batch_i][i1][1] == 0
                    ):
                        break
                    wordpiece_absolute_indices.append(
                        wordpiece_indices[batch_i][i1][0] - reduce_offset
                    )
                    reduce_offset += (
                        wordpiece_indices[batch_i][i1][1]
                        - wordpiece_indices[batch_i][i1][0]
                        - 1
                    )

                for i1 in sent_i_b_nps:
                    res = "".join(
                        random.choices(string.ascii_uppercase + string.digits, k=25)
                    )
                    if i1 in wordpiece_absolute_indices:
                        if subwords.shape[0] == 0:
                            subwords = sent_i[i1].cpu().numpy()
                        else:
                            subwords = np.vstack((subwords, sent_i[i1].cpu().numpy()))
                        # print('subwords',subwords)
                        # input()
                        # torch.save(sent_i[i1].cpu().numpy(),'data/tsne/B_NP/'+str(self.hyper_params['which_way'])+'/subwords/'+res+'.bin')
                    else:
                        if nonsubwords.shape[0] == 0:
                            nonsubwords = sent_i[i1].cpu().numpy()
                        else:
                            nonsubwords = np.vstack(
                                (nonsubwords, sent_i[i1].cpu().numpy())
                            )
                        # print('nonsubwords',nonsubwords)
                        # input()
                        # torch.save(sent_i[i1].cpu().numpy(),'data/tsne/B_NP/'+str(self.hyper_params['which_way'])+'/nonsubwords/'+res+'.bin')
            numpy_containers[0] = subwords
            numpy_containers[1] = nonsubwords

        out = self.fc1(out)  # [batches, max_len, len(tag_set)]
        logits = self.activation(out)  # [batches, max_len, len(tag_set)]
        y = y.float().view(-1, out.shape[2])
        y = torch.argmax(y, dim=1)
        active_y = torch.where(
            attention_mask.view(-1) == 1,
            y.view(-1),
            torch.tensor(self.criterion.ignore_index).type_as(y),
        )
        loss = self.criterion(logits.view(-1, out.shape[2]), active_y)
        return SequenceClassifierOutput(loss=loss, logits=logits)

    def forward_for_prediction(self, input_ids, attention_mask, wordpiece_indices):
        if self.hyper_params["embedding_way"] == "last_hidden_state":
            out = self.model(
                input_ids, attention_mask=attention_mask
            ).last_hidden_state  # [batches, max_len, 768]
        elif self.hyper_params["embedding_way"] == "first_two":
            out = self.model(input_ids, attention_mask=attention_mask)
            out = self.take_mean([out.hidden_states[0], out.hidden_states[1]])
        elif self.hyper_params["embedding_way"] == "last_two":
            out = self.model(input_ids, attention_mask=attention_mask)
            out = self.take_mean([out.hidden_states[-2], out.hidden_states[-3]])
        else:
            raise "Unknown embedding_way specified"
        if self.hyper_params["which_way"] == 3:
            c = torch.zeros(out.shape).to(self.device)
            for i in range(out.shape[0]):
                out_sent_i = out[i]
                wpi_sent_i = wordpiece_indices[i]
                wpi_sent_i = (wpi_sent_i[wpi_sent_i != 0]).tolist()
                # print('$$',wpi_sent_i)
                j, k = 0, 0
                while j < out_sent_i.shape[0]:
                    if j not in wpi_sent_i[::2]:
                        c[i, k] += out[i, j]
                        j += 1
                    else:
                        start = j
                        end = wpi_sent_i[2 * wpi_sent_i[::2].index(j) + 1]
                        c[i, k] += torch.mean(out[i][start:end], 0)
                        j = end
                    k += 1
            out = c
        # out, _ = self.rnn1(out) # [batches, max_len, len(tag_set)]
        out = self.fc1(out)  # [batches, max_len, len(tag_set)]
        logits = self.activation(out)  # [batches, max_len, len(tag_set)]
        return SequenceClassifierOutput(logits=logits)


def predict_with_model(model, sent, tokenizer):
    model.eval()
    a = tokenizer("i", return_tensors="pt", max_length=4, padding="max_length")[
        "input_ids"
    ][0]
    cls_id, sep_id, pad_id = a[0], a[2], a[3]

    input_ids = []
    attention_mask = []
    tag_mask = []
    wordpiece_indices = []
    max_len = model.hyper_params["max_len"]

    tii, wi = [cls_id], []
    for i, (word) in enumerate(sent.split("\t")):
        tids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(word))
        if len(tids) > 1:
            if (
                model.hyper_params["which_way"] == 3
            ):  # average out the wordpiece embeddings during the forward pass
                wi.append([len(tii), len(tii) + len(tids)])
            elif model.hyper_params["which_way"] == 2:  # take last wordpiece token id
                tids = [tids[-1]]
            elif model.hyper_params["which_way"] == 1:  # take first wordpiece token id
                tids = [tids[0]]
        tii += tids

    tii.append(sep_id)
    tam = [1] * len(tii) + [0] * (max_len - len(tii))
    tm = (
        [0]
        + [1] * (len(sent.split("\t")))
        + [0] * (max_len - len(sent.split("\t")) - 1)
    )
    if len(tii) > 511:
        return ["Size exceeded"]
    tii = tii + [pad_id] * (max_len - len(tii))
    input_ids.append(tii)
    attention_mask.append(tam)
    tag_mask.append(tm)
    wi = wi + [[0, 0]] * (
        max_len - len(wi)
    )  # putting max_len is a bit of overkill here I know. Length of wi represents the number of words in the sentence that can be splitted into multiple wordpieces. I wrote max_len here just for consistency.
    wordpiece_indices.append(wi)

    input_ids = torch.tensor(input_ids).to(model.device)
    attention_mask = torch.tensor(attention_mask).to(model.device)
    tag_mask = torch.tensor(tag_mask).to(model.device)
    wordpiece_indices = torch.tensor(wordpiece_indices).to(model.device)

    with torch.no_grad():
        out = model.forward_for_prediction(
            input_ids, attention_mask, wordpiece_indices
        ).logits

    y_pred = model.predict_from_logits(out, tag_mask)
    return y_pred
