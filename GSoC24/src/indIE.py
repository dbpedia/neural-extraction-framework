from chunking.chunking_model import *
from chunking.crf_chunker import *
from utils import *
import pandas as pd
from tqdm import tqdm
import stanza, torch

use_gpu = torch.cuda.is_available()
hindi_nlp = load_stanza_model(lang="hi", use_gpu=use_gpu)

nlp = hindi_nlp
lang = "hi"

hyper_params = {
    "run_ID": 26,
    "createData": True,
    "bs": 8,
    "bert_model_name": "xlm-roberta-base",
    "available_models": "'ai4bharat/indic-bert' 'bert-base-multilingual-cased' 'xlm-roberta-base' 'bert-base-multilingual-uncased' 'xlm-mlm-100-1280' 'xlm-mlm-tlm-xnli15-1024' 'xlm-mlm-xnli15-1024'",
    "alpha": 0.00001,
    "epochs": 3,
    "rseed": 123,
    "nw": 4,
    "train_ratio": 0.7,
    "val_ratio": 0.1,
    "max_len": 275,
    "which_way": 3,
    "which_way_gloss": "1= take first wordpiece token id | 2= take last wordpiece token id | 3= average out the wordpiece embeddings during the forward pass",
    "embedding_way": "last_hidden_state",
    "embedding_way_gloss": "last_hidden_state, first_two, last_two",
    "notation": "BI",
    "platform": 1,
    "available_platforms": "MIDAS server = 1, colab = 2",
    "chunker": "XLM",  # CRF or XLM
}

model_embeddings = {
    "ai4bharat/indic-bert": 768,
    "bert-base-multilingual-cased": 768,
    "xlm-roberta-base": 768,
    "bert-base-multilingual-uncased": 768,
    "xlm-mlm-100-1280": 1280,
    "xlm-mlm-tlm-xnli15-1024": 1024,
    "xlm-mlm-xnli15-1024": 1024,
}

hyper_params["embedding_size"] = model_embeddings[hyper_params["bert_model_name"]]
my_tagset = torch.load("input/my_tagset_" + hyper_params["notation"] + ".bin")
hyper_params["my_tagset"] = my_tagset

os.environ["PYTHONHASHSEED"] = str(hyper_params["rseed"])
# Torch RNG
torch.manual_seed(hyper_params["rseed"])
torch.cuda.manual_seed(hyper_params["rseed"])
torch.cuda.manual_seed_all(hyper_params["rseed"])
# Python RNG
np.random.seed(hyper_params["rseed"])
random.seed(hyper_params["rseed"])

is_cuda = torch.cuda.is_available()
if is_cuda and use_gpu:
    device = torch.device("cuda:0")
    t = torch.cuda.get_device_properties(device).total_memory
    c = torch.cuda.memory_reserved(device)
    a = torch.cuda.memory_allocated(device)
    f = t - (c - a)  # free inside cache
    print(
        "GPU is available",
        torch.cuda.get_device_name(),
        round(t / (1024 * 1024 * 1024)),
        "GB",
    )
else:
    device = torch.device("cpu")
    print("GPU not available, CPU used")

hyper_params["device"] = str(device)

if hyper_params["chunker"] == "XLM":
    print("Creating the XLM chunker model...")
    model = chunker_class(device, hyper_params).to(device)
    checkpoint = torch.load(
        "models/state_dicts/model/" + str(hyper_params["run_ID"]) + "_epoch_4.pth.tar",
        map_location=device,
    )
    checkpoint["state_dict"].pop("model.embeddings.position_ids")
    print(model.load_state_dict(checkpoint["state_dict"]))
    tokenizer = AutoTokenizer.from_pretrained(hyper_params["bert_model_name"])
elif hyper_params["chunker"] == "CRF":
    model, tokenizer = "CRF", "CRF"


def get_triples(sentence):
    start_time = time.time()
    total_exts = []
    all_sents, exts, ctime, etime = perform_extraction(
        sentence, lang, model, tokenizer, nlp, show=False
    )  # show = True to display intermediate results
    # print(all_sents, exts, ctime, etime, sep="\n\n")
    # print(augument_extractions(exts))
    if len(exts) == 0:
        print("zero_extraction")
    else:
        total_exts.extend(exts)
    # ctags = predict_with_model(model,s,tokenizer)
    ttaken = round(time.time() - start_time, 3)
    # print(ttaken)
    return total_exts, ttaken
