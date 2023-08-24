import ast
import torch
import pandas as pd
from nltk import sent_tokenize

gen_kwargs = {
    "max_length": 256,
    "length_penalty": 0,
    "num_beams": 10,
    "num_return_sequences": 1,
}


# Function to parse the generated text and extract the triplets
def extract_triplets(text):
    triplets = []
    relation, subject, relation, object_ = "", "", "", ""
    text = text.strip()
    current = "x"
    for token in (
        text.replace("<s>", "").replace("<pad>", "").replace("</s>", "").split()
    ):
        if token == "<triplet>":
            current = "t"
            if relation != "":
                triplets.append(
                    {
                        "head": subject.strip(),
                        "type": relation.strip(),
                        "tail": object_.strip(),
                    }
                )
                relation = ""
            subject = ""
        elif token == "<subj>":
            current = "s"
            if relation != "":
                triplets.append(
                    {
                        "head": subject.strip(),
                        "type": relation.strip(),
                        "tail": object_.strip(),
                    }
                )
            object_ = ""
        elif token == "<obj>":
            current = "o"
            relation = ""
        else:
            if current == "t":
                subject += " " + token
            elif current == "s":
                object_ += " " + token
            elif current == "o":
                relation += " " + token
    if subject != "" and relation != "" and object_ != "":
        triplets.append(
            {"head": subject.strip(), "type": relation.strip(), "tail": object_.strip()}
        )
    return triplets


def extract_relations_rebel(model, tokenizer, text):
    """Return the annotated text(with annotations for triplet, subject, object etc).

    Args:
        model: The REBEL model loaded from HF.
        tokenizer: The tokenizer for REBEL model loaded from HF.
        text: The text to annotate.

    Returns:
    Triplet annotated text.

    ```
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("Babelscape/rebel-large")
    model = AutoModelForSeq2SeqLM.from_pretrained("Babelscape/rebel-large")
    ```
    """

    tokenized_sentences = sent_tokenize(text)
    list_triples = []

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    for text in tokenized_sentences:
        model_inputs = tokenizer(
            text, max_length=256, padding=True, truncation=True, return_tensors="pt"
        )
        # Generate
        generated_tokens = model.generate(
            model_inputs["input_ids"].to(model.device),
            attention_mask=model_inputs["attention_mask"].to(model.device),
            **gen_kwargs,
        )

        decoded_preds = tokenizer.batch_decode(
            generated_tokens, skip_special_tokens=False
        )

        l1 = []
        for idx, sentence in enumerate(decoded_preds):
            l1 += extract_triplets(sentence)

        d1 = {}
        ctr = 0

        for x in l1:
            ctr += 1
            if not str(x) in d1:
                d1[str(x)] = 0
            d1[str(x)] += 1

        for x in d1:
            t = x.replace("}", "")
            final_dict = t + ", 'Confidence': " + str(d1[x] / ctr) + "}"
            final_dictionary = ast.literal_eval(final_dict)
            list_triples.append(final_dictionary)

    return pd.DataFrame(list_triples)
