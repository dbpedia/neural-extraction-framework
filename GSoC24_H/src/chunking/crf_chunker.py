import subprocess
import time
import os
import sklearn_crfsuite, pickle


def word2features(sent, i):
    word = sent[i][0]
    postag = sent[i][1]

    features = {
        "bias": 1.0,
        "word": word,
        "postag": postag,
        "-2:postag": sent[i - 2][1] if i - 2 >= 0 else "X",
        "-1:postag": sent[i - 1][1] if i - 1 >= 0 else "X",
        "+1:postag": sent[i + 1][1] if i + 1 < len(sent) else "X",
        "+2:postag": sent[i + 2][1] if i + 2 < len(sent) else "X",
    }

    return features


def sent2features(sent):
    return [word2features(sent, i) for i in range(len(sent))]


def sent2labels(sent):
    return [label for token, postag, label in sent]


def sent2tokens(sent):
    return [token for token, postag, label in sent]


def reduce_one_dim(a):
    flat_list = [item for sublist in a for item in sublist]
    # b = []
    # for c in tqdm(a, desc='reducing dimension'):
    # 	b = b + c
    return flat_list


def predict_with_crf(sent):
    test_sent = []

    for word in sent.words:
        test_sent.append((word.text, word.upos))

    test_sent = [test_sent]
    X_test = [sent2features(s) for s in test_sent]

    crf = sklearn_crfsuite.CRF(
        algorithm="lbfgs",
        c1=1.3518036184801792,  # obtained from grid search
        c2=0.04058208879576922,
        max_iterations=1000,
        all_possible_transitions=True,
        verbose=True,
    )

    file = open(
        "chunking/state_dicts/model/sklearn_crf_model_v2_pos_mapped_2.pkl", "rb"
    )
    crf = pickle.load(file)
    file.close()

    y_pred = reduce_one_dim(crf.predict(X_test))
    return y_pred

    print(y_pred)

    exit()

    # doc = nlp(sent)
    wordl = []
    posl = []
    for sentence in doc.sentences:
        for word in sentence.words:
            wordl.append(word.text)
            posl.append(word.upos)
    assert len(wordl) == len(posl)
    file = open("tempfile.txt", "w")
    for w, p in zip(wordl, posl):
        file.write(w + "\t" + p + "\tX\n")
    file.close()
    # bashCommand = "/media/data_dump/Ritwik/crf/CRF++-0.58/crf_test -m /media/data_dump/Ritwik/crf/CRF++-0.58/example/chunking/crf_model tempfile.txt > tempfile2.txt"
    # process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
    # output, error = process.communicate()
    process = subprocess.Popen(
        "/media/data_dump/Ritwik/crf/CRF++-0.58/crf_test -m /media/data_dump/Ritwik/crf/CRF++-0.58/example/chunking/crf_model tempfile.txt > tempfile2.txt",
        shell=True,
    )
    process.wait()
    # time.sleep(5)
    os.remove("tempfile.txt")
    file = open("tempfile2.txt", "r")
    content = [x.strip() for x in file.readlines()]
    file.close()
    ml = []
    for line in content:
        if line:
            ml.append(line.split("\t")[-1])
    os.remove("tempfile2.txt")
    return ml
