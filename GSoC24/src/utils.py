import nltk
from nltk import Tree
import numpy as np
import stanza, time
from collections import defaultdict
from chunking.chunking_model import *
from chunking.crf_chunker import *
import colorsys
import random
import graphviz
import subprocess, importlib

stanza_path = importlib.util.find_spec("stanza").submodule_search_locations[0]
# https://stackoverflow.com/questions/269795/how-do-i-find-the-location-of-python-module-sources
# https://stackoverflow.com/questions/35288021/what-is-the-equivalent-of-imp-find-module-in-importlib
stanza_version = stanza.__version__

if not os.path.exists(stanza_path + "/" + stanza_version + "/"):
    os.makedirs(stanza_path + "/" + stanza_version + "/")

prep_dict = {"में": "in"}
action_relations = ["root", "aux", "cop", "mark", "advcl", "acl", "acl:relcl"]
arg_deprels = ["subj", "comp", "obj", "obl", "nmod", "appos", "nummod"]


def to_nltk_tree(mat, n, msg=""):
    """
    this function converts dependency matrix to dependency tree
    """
    children = mat.children(n)
    if len(children) != 0:
        return Tree(
            mat.node_text(n, msg),
            [to_nltk_tree(mat, int(child.id) - 1, msg) for child in children],
        )
    else:
        return mat.node_text(n, msg)


class phrases:
    def __init__(self, s, h, i, upos, xpos, deprellist, deprel, ptype):
        """
        this class is to create spacy like class for phrases
        """
        (
            self.text,
            self.head,
            self.id,
            self.upos,
            self.xpos,
            self.deprellist,
            self.deprel,
            self.ptype,
            self.state,
        ) = (s, h, i, upos, xpos, deprellist, deprel, ptype, "")

    def values(self):
        return (
            self.text
            + "|"
            + str(self.head)
            + "|"
            + str(self.id)
            + "|"
            + self.upos
            + "|"
            + self.deprellist
            + "|"
            + str(self.deprel)
            + "|"
            + self.ptype
        )


class dmatrix:
    def __init__(self, ml):
        """
        if cell ij has 1, then it indicates that word j is a child of word i. Hence parents are stored in rows, and children are stored in columns.
        """
        self.words = ml
        self.mat = np.zeros((len(ml), len(ml)))
        self.root_value = 0
        for d in ml:
            if d.head != 0:
                self.mat[d.head - 1, int(d.id) - 1] = 1
            elif d.head == 0 and str(d.deprel) == "root":
                self.root_value = d.id - 1

        # self.root_value = self.mat.sum(axis=0).argmin()

    def __len__(self):
        return len(self.words)

    def show(self):
        """
        shows the dmatrix
        """
        print("    ", end="")
        for i in range(len(self.words)):
            print((i + 1) % 10, end="    ")
        print()
        for i in range(len(self.words)):
            print((i + 1) % 10, list(self.mat[i]))

    def get_root(self):
        return self.root_value

    def set_root(self, n):
        self.root_value = n
        return

    def children(self, n):
        # returns a list of children
        c_indexList = np.where(self.mat[n] == 1)[0]
        children = []
        for cindx in c_indexList:
            children.append(self.words[int(cindx)])
        return children

    def parent(self, n):
        # returns parent
        if self.root_value == n:
            return -1
        else:
            return self.words[np.where(self.mat[:, n] == 1)[0][0]]

    def text(self, n):
        return self.words[n].text

    def node_text(self, n, msg=""):
        """
        this function returns the text that is to be displayed on the tree nodes
        """
        w = self.words[n]
        s = (
            "^"
            + str(w.id)
            + "_"
            + str(w.text)
            + "&"
            + str(w.xpos)
            + "@"
            + str(w.upos)
            + "%"
            + str(w.deprel)
            + "$"
        )
        s = w.text + "_" + str(w.deprel)
        if msg == "mpt":
            s = str(w.id - 1) + "_" + w.text + "\n" + str(w.deprel) + "\n" + str(w.upos)
        elif msg == "urdu":
            s = str(w.id - 1) + "\n" + str(w.deprel) + "\n" + str(w.upos)
        else:
            s = (
                "  "
                + str(w.id - 1)
                + "_"
                + str(w.text)
                + "  "
                + "\n"
                + str(w.deprel)
                + "\n"
                + str(w.upos)
            )
        return s

    def n_descendants(self, n):
        """
        this calculates number of nodes which has this given node as an ancestor
        or in simple language
        number of possible children/grand children/great grand children etc of this node
        """
        a = len(self.children(n))
        child_queue = [int(child.id) - 1 for child in self.children(n)]
        while len(child_queue) > 0:
            tl = [int(child.id) - 1 for child in self.children(child_queue[0])]
            a = a + len(tl)
            child_queue = child_queue + tl
            child_queue.reverse()
            child_queue.pop()
            child_queue.reverse()
        return a

    def all_descendants(self, n):
        child_queue = [int(child.id) - 1 for child in self.children(n)]
        descendants = [int(child.id) - 1 for child in self.children(n)]
        while len(child_queue) > 0:
            tl = [int(child.id) - 1 for child in self.children(child_queue[0])]
            descendants += tl
            child_queue = child_queue + tl
            child_queue.reverse()
            child_queue.pop()
            child_queue.reverse()
        return descendants

    def delete_node(self, n):
        children = self.children(n)
        parent = self.parent(n)
        if parent != -1:
            self.mat[parent.id - 1, self.words[n].id - 1] = 0
            self.mat[parent.id - 1] += self.mat[self.words[n].id - 1]
            for child in children:
                child.head = parent.id
                self.mat[n, child.id - 1] = 0
                self.mat[parent.id - 1, child.id - 1] = 1
        else:
            # for word in self.words:
            #     print(word.id-1,word.text)
            print(self.words[n].values())
            raise "Cannot delete root node"
        return

    def siblings(self, n):
        parent = self.words[n].head - 1
        # print('parent is',self.words[parent].text)
        # input()
        tempmat = self.mat.copy()
        assert (
            tempmat[parent, n] == 1
        )  # make sure this node and its parent are connected
        tempmat[parent, n] = 0  # now break the connection
        c_indexList = np.where(tempmat[parent] == 1)[0]  # find children of its parent
        siblingList = []
        for cindx in c_indexList:
            siblingList.append(self.words[int(cindx)])
        return siblingList


def fixable(tag1, tag2, show=False):
    return False
    if "VG" in tag1 and "VG" in tag2:
        pp("\tFIXED", show)
        return True
    else:
        pp("\tCANNOT BE FIXED", show)
        return False


def predicted_ctag_validity(ctags, show=False):
    if "I_" in ctags[0]:
        pp("\tI in starting", show)
        return False
    i = 0
    while i < len(ctags) - 1:
        if "I_" in ctags[i] and "I_" in ctags[i + 1]:
            if ctags[i].replace("I_", "") != ctags[i + 1].replace("I_", ""):
                pp("\tI_X I_Y observed", show)
                pp("\t" + "\t".join(ctags[i - 1 : i + 2]), show)
                return fixable(
                    ctags[i].replace("I_", ""), ctags[i + 1].replace("I_", ""), show
                )
        if "B_" in ctags[i] and "I_" in ctags[i + 1]:
            if ctags[i].replace("B_", "") != ctags[i + 1].replace("I_", ""):
                pp("\tB_X I_Y observed", show)
                pp("\t" + "\t".join(ctags[i - 1 : i + 2]), show)
                return fixable(
                    ctags[i].replace("B_", ""), ctags[i + 1].replace("I_", ""), show
                )
        i += 1
    return True


def resolve_Xs(ctag):
    for i, ct in enumerate(ctag):
        if ct == "X":
            if i != 0:
                ctag[i] = (
                    "I_" + ctag[i - 1].split("_")[1]
                )  # 'I_' + previous chunk tag label
            else:
                ctag[i] = (
                    "B_" + ctag[i + 1].split("_")[1] if i + 1 < len(ctag) else "NP"
                )  # 'B_' + next chunk tag label
    return ctag


def augument_extractions(exts, prev=[], repeat=False):
    aug = []
    for s_exts in exts:
        taug, pairs = [], []
        d = defaultdict(bool, {})
        for i, e1 in enumerate(s_exts):
            for j, e2 in enumerate(s_exts):
                if (
                    len(set(e1).union(set(e2))) == 4
                    and e1[1] == e2[1]
                    and (j, i) not in pairs
                    and d[tuple(set(e1).union(set(e2)))] == False
                ):
                    head = list(set(e1) - (set(e1).intersection(set(e2))))[0]
                    rel = e1.copy()
                    rel.remove(head)
                    rel = " ".join(rel)
                    tail = list(set(e2) - (set(e1).intersection(set(e2))))[0]
                    taug.append([head, rel, tail])
                    pairs.append((i, j))
                    d[tuple(set(e1).union(set(e2)))] = 1
                    # print(tuple(set(e1).union(set(e2))))
        if repeat:
            pl = []
            for p in pairs:
                pl.append(p[0])
                pl.append(p[1])
            for i in range(len(s_exts)):
                if i not in pl:
                    taug.append(s_exts[i])
        aug.append(taug)

    return aug  # comment this line to make the process recursive
    if aug == prev:
        # print(aug)
        # input('1')
        return aug
    else:
        # print(aug)
        # input('2')
        aug = augument_extractions(aug, aug, True)
        return aug


def perform_extraction(
    s, lang, model, tokenizer, nlp, show=False, argshow=False, is_a_override=False
):
    if lang == "hi":
        s = s.replace("|", "।")
    s = re.sub(r"[\U0001F004-\U0001FA95]+", "", s)  # emoji removal
    s = re.sub(r"[\uFE00-\uFE0F]+", "", s)  # some special character removal
    doc = nlp(s)
    all_sents, exts = [], []
    chunking_time, ext_time = [], []
    #                      hai       oru       undi
    is_a = {"hi": "है", "ur": "ہے", "ta": "ஒரு", "te": "ఉంది", "mr": "आहे", "en": "is"}[
        lang
    ]  # these are the language specific is_a labels
    for sent in doc.sentences:
        pp("Orig:" + s, show)
        pp("Curr:" + sent.text, show)
        ml = []
        for word in sent.words:
            ml.append(word)
        m = dmatrix(ml)  # <------------- creating traditional dependency tree
        c = "\t".join([w.text for w in ml])
        pp(c, show)
        a = time.time()
        if model == "CRF":
            ctags = predict_with_crf(sent)
        else:
            ctags = predict_with_model(model, c, tokenizer)
        if ctags[0] == "Size exceeded":
            pp("Size exceeded for chunking", show)
            all_sents.append(sent.text)
            exts.append([])
            chunking_time.append(0)
            ext_time.append(0)
            continue

        ctags = resolve_Xs(ctags)
        pp("\t".join(ctags), show)
        if not predicted_ctag_validity(ctags, show):
            pp("Error in chunk tags", show)
            all_sents.append(sent.text)
            exts.append([])
            chunking_time.append(0)
            ext_time.append(0)
            continue
        w_idx_to_p_id = {}
        cid = -1
        textL, headL, idL, uposL, xposL, deprelL, phraseL = [], [], [], [], [], [], []
        # ----------------------- collecting all phrases in a list ---------------------
        for i, ct in enumerate(ctags + ["B_END"]):
            if (
                "B_" in ct
            ):  # notice: this mechanism is independent of chunk labels, it only looks for chunk boundaries
                cid += 1
                if textL:
                    ph = phrases(
                        " ".join(textL),
                        0,
                        cid,
                        ".".join(uposL),
                        ".".join(xposL),
                        ".".join(deprelL),
                        0,
                        ptype,
                    )  # though it stores the chunk label (ptype) in case it is needed in future
                    phraseL.append(ph)
                textL, headL, idL, uposL, xposL, deprelL = [], [], [], [], [], []
            if i < len(ctags):
                textL.append(ml[i].text)
                uposL.append(ml[i].upos)
                xposL.append(ml[i].xpos)
                deprelL.append(str(ml[i].deprel))
                ptype = ct[2:]
                w_idx_to_p_id[i] = cid + 1  # <------------------ word to phrase mapping
        chunking_time.append(round(time.time() - a, 3))
        pp("Chunking happened in " + str(time.time() - a) + " secs", show)
        # -------------------------------------------------------------------------------
        a = time.time()
        # for p in phraseL:
        #     print(p.values(), end=' -- ')
        # print()
        # print(w_idx_to_p_id)

        # phrase List me phrases ke index 1 se start hote hai
        #          bcoz phrase ke head bhi 1 se start hote hai bcoz stanza me aisa hota tha, and dmatrix is designed in such a way
        # stanza me har word ki ID 1 se start hoti hai

        # Indices of phrases in phraseList starts at 1
        # 			because head of a phrase starts at 1 as well
        # 				because stanza has this kind of format (where id of root is given 1, and its head is given 0), and dmatrix is designed in such a way

        # m.show()

        # ------------------------- filling the appropriate parent ids using dependency tree ---------------------
        # ml[m.get_root()-1]
        phrases_included = [w_idx_to_p_id[m.get_root()] - 1]
        phraseL[phrases_included[0]].deprel = "root"
        # print('m.get_root()',m.get_root(), 'w_idx_to_p_id[m.get_root()]', w_idx_to_p_id[m.get_root()], 'phrase_included',phrases_included)
        # print(m.children(5))
        children_queue = m.children(m.get_root())
        i = 0
        while i < len(children_queue):
            child = children_queue[i]
            if not (str(child.deprel) == "punct" and child.upos == "PUNCT"):
                if w_idx_to_p_id[child.id - 1] - 1 not in phrases_included:
                    word_id_0_indexing = child.id - 1
                    phrase_id_0_indexing = w_idx_to_p_id[word_id_0_indexing] - 1
                    phrases_included.append(phrase_id_0_indexing)
                    phraseL[phrase_id_0_indexing].head = w_idx_to_p_id[child.head - 1]
                    phraseL[phrase_id_0_indexing].deprel = str(child.deprel)
                nxt_children = m.children(child.id - 1)
                if nxt_children:
                    children_queue += nxt_children
            i += 1
        # -----------------------------------------------------------------------------------------------------------

        # ------------------ printing traditional dependency tree -------------
        if lang == "ur":
            if show:
                for sents in doc.sentences:
                    for i, word in enumerate(sents.words):
                        print(i, word.text, word.upos, word.deprel)
            tree = to_nltk_tree(m, m.get_root(), "urdu")
        else:
            tree = to_nltk_tree(m, m.get_root())
        if show and type(tree) == nltk.tree.Tree:
            tree.pretty_print()
        # ----------------------------------------------------------------------

        # for p in phraseL:
        #     print(p.values(), end=' -- ')
        # print('\n')
        m = dmatrix(phraseL)  # <------------- creation of phrase-level dependency tree

        # ------------------ adjustments in phrase-level dependency tree -------------------
        i = 0
        while i < len(phraseL):
            if str(phraseL[i].deprel) in ["compound"]:
                phraseL[phraseL[i].head - 1].text = (
                    phraseL[i].text + " " + phraseL[phraseL[i].head - 1].text
                )
                phraseL[phraseL[i].head - 1].upos = (
                    phraseL[i].upos + "." + phraseL[phraseL[i].head - 1].upos
                )
                phraseL[phraseL[i].head - 1].xpos = (
                    phraseL[i].xpos + "." + phraseL[phraseL[i].head - 1].xpos
                )
                phraseL[phraseL[i].head - 1].deprellist = (
                    phraseL[i].deprellist
                    + "."
                    + phraseL[phraseL[i].head - 1].deprellist
                )
                phraseL[i].text, phraseL[i].deprel, phraseL[i].upos = "--", "--", "--"
            elif str(phraseL[i].deprel) in ["aux", "advmod", "case"]:
                phraseL[phraseL[i].head - 1].text = (
                    phraseL[phraseL[i].head - 1].text + " " + phraseL[i].text
                )
                phraseL[phraseL[i].head - 1].upos = (
                    phraseL[phraseL[i].head - 1].upos + "." + phraseL[i].upos
                )
                phraseL[phraseL[i].head - 1].xpos = (
                    phraseL[phraseL[i].head - 1].xpos + "." + phraseL[i].xpos
                )
                phraseL[phraseL[i].head - 1].deprellist = (
                    phraseL[phraseL[i].head - 1].deprellist
                    + "."
                    + phraseL[i].deprellist
                )
                phraseL[i].text, phraseL[i].deprel, phraseL[i].upos = "--", "--", "--"
            i += 1

        # for p in phraseL:
        #     print(p.values(), end=' -- ')
        # print('\n','this '*20)

        m = dmatrix(phraseL)
        for i in range(len(m)):
            if str(m.words[i].deprel) == "--":
                m.delete_node(i)
        # ----------------------------------------------------------------------------------

        # ------------------ printing phrase-level dependency tree -------------
        if lang == "ur":
            if show:
                for i, phrase in enumerate(phraseL):
                    print(i, phrase.text, phrase.upos, phrase.deprel)
            tree = to_nltk_tree(m, m.get_root(), "urdu")
        else:
            tree = to_nltk_tree(m, m.get_root(), "mpt")
        if show and type(tree) == nltk.tree.Tree:
            tree.pretty_print()
        # show=False
        # ----------------------------------------------------------------------

        clean_state(m)
        exts1 = extract(
            m, m.get_root(), [], is_a, 0, show, argshow, is_a_override
        )  # <---------- Information Extraction happens here
        exts2 = []
        for e in exts1:  # removing duplicate extractions
            if e not in exts2:
                exts2.append(e)
        exts.append(exts2)
        all_sents.append(sent.text)
        ext_time.append(round(time.time() - a, 3))
        pp("Extraction happened in " + str(time.time() - a) + " secs", show)
    return all_sents, exts, chunking_time, ext_time


def pp(msg, show, endc="\n"):
    if show:
        print(msg, end=endc)
    return


def clean_state(m):
    for i in range(len(m)):
        m.words[i].state = ""
    return


def closest_phrase(m, phrase1, phraseN):
    pl = [p.text for p in m.words] if type(m) == dmatrix else m.split(" ")
    pl_d = []
    for p in phraseN:
        try:
            a = pl.index(phrase1)
            b = pl.index(p)
            d = abs(a - b)
        except Exception as e:
            try:
                a = " ".join(pl).index(phrase1)
                b = " ".join(pl).index(p)
            except Exception as e:
                a = " ".join(pl).index(phrase1.split()[0])
                b = " ".join(pl).index(p.split()[0])
            offset = len(phrase1) if a < b else len(p)
            d = abs(a - b) - offset + 1
        pl_d.append(d)
    return phraseN[np.argmin(pl_d)]


def obl_useful(m, c, stateless=False):
    if c.head != 0:  # means c is not root
        parent = m.words[c.head - 1]
        if (
            str(parent.deprel) == "amod"
            and parent.ptype == "JJP"
            and "PROPN" not in c.upos
        ):
            c.state = "done" if not stateless else c.state
            return False
        elif (
            str(parent.deprel) == "root"
            and parent.upos == "VERB"
            and "PROPN" not in c.upos
        ):
            c.state = "done" if not stateless else c.state
            return False
        elif str(parent.deprel) == "obl" and "PROPN" not in c.upos:
            c.state = "done" if not stateless else c.state
            return False
    if c.upos == "PRON":
        c.state = "done" if not stateless else c.state
        return False
    return True


def nmod_useful(m, c, stateless=False):
    # if list(set(c.upos.split('.'))) == ['NOUN']: # if it has only nouns
    #    return False

    parent = m.words[c.head - 1]
    if (
        "subj" in str(parent.deprel)
        or "obj" in str(parent.deprel)
        or str(parent.deprel) == "root"
        or (str(parent.deprel) == "obl" and obl_useful(m, parent, stateless=stateless))
        or if_any_in(action_relations, str(parent.deprel))
    ):
        return True
    if "PROPN" in c.upos:
        return True
    c.state = "done" if not stateless else c.state
    return False
    return True


def find_args(m, n, patterns, show, stateless=False):
    """
    stateless: states change mat karna, I am just testing the function
    """
    arg = phrases("", -1, -1, "", "", "", "", "")
    for p in patterns:

        if len(p) == 2:
            if not arg.text:
                for c in m.children(n):
                    if (
                        p[0] in str(c.deprel)
                        and p[1](m, c, stateless=stateless)
                        and "done" not in c.state
                    ):
                        arg = c
                        c.state = "done" if not stateless else c.state
                        pp("\n\t[" + c.text + "] is a <" + p[0] + ">", show)
                        return arg
                    else:
                        pp(
                            "\t["
                            + c.text
                            + "] is not a <"
                            + p[0]
                            + "> because "
                            + str(p[0] in str(c.deprel))
                            + str(p[1](m, c, stateless=True))
                            + str("done" not in c.state)
                            + "#___# ",
                            show,
                            "",
                        )

        else:
            if not arg.text:
                for c in m.children(n):
                    if p[0] in str(c.deprel) and "done" not in c.state:
                        arg = c
                        if not stateless:
                            c.state = "done"
                        pp("\n\t[" + c.text + "] is a <" + p[0] + ">", show)
                        return arg
                    else:
                        pp(
                            "\t["
                            + c.text
                            + "] is not a <"
                            + p[0]
                            + "> because "
                            + str(p[0] in str(c.deprel))
                            + str("done" not in c.state)
                            + "#___# ",
                            show,
                            "",
                        )
    pp("", show)
    return arg


def clausal_node(m, n):
    head = m.words[n].text.split()[m.words[n].upos.split(".").index("PRON")]
    rel = m.words[n].text.split()[m.words[n].upos.split(".").index("VERB")]
    if "PART" in m.words[n].upos:
        rel = (
            m.words[n].text.split()[m.words[n].upos.split(".").index("PART")]
            + " "
            + rel
        )
    if "AUX" in m.words[n].upos:
        if m.words[n].upos.split(".").count("AUX") == 1:
            rel = (
                rel
                + " "
                + m.words[n].text.split()[m.words[n].upos.split(".").index("AUX")]
            )
        else:
            for idx in [
                i for i, x in enumerate(m.words[n].upos.split(".")) if x == "AUX"
            ]:
                rel = rel + " " + m.words[n].text.split()[idx]
    tail = m.words[n].text.split()
    tail.remove(head)
    for r in rel.split(" "):
        tail.remove(r)
    tail = " ".join(tail)
    return head, rel, tail


def is_clausal_node(m, n):
    return (
        "PRON" in m.words[n].upos.split(".")
        and (
            "NOUN" in m.words[n].upos.split(".")
            or "PROPN" in m.words[n].upos.split(".")
        )
        and ("VERB" in m.words[n].upos.split("."))
    )


def if_any_in(wlist, w):
    for w2 in wlist:
        if w2 in w:
            return True
    return False


def extract(
    m,
    n,
    extraction_queue,
    is_a,
    running_no=0,
    show=False,
    argshow=False,
    is_a_override=False,
):
    """
    m: dmatrix
            It is a matrix representation of MDT (refer paper to know MDT)
    n: Integer
            Id of the phrase you are standing on right now. It starts with root.
    extraction_queue: list of lists
            as the name suggests, it is a queue that stores the extractions
    running_no: Integer
            a number which helps us to infinite loops, during recursion (never happened in expts)
    show: Bool
            A flag to show the intermediate steps in an abstract level.
    argshow: Bool
            it is a flag to show the intermediate steps while matching the arguments. It can be said that it is used when we want to see most fine-grained intermediate steps.
    is_a_override: Bool
            This flag is set to True only in case of Hindi Benchie. It uses 'property' instead of language specific is_a labels
    """
    pp("-" * 50, show)
    pp("\t\t Node=" + m.words[n].text + "\n", show)

    if "cop" in [str(x.deprel) for x in m.children(n)]:
        pp('Copular verb found in the children of "' + m.words[n].text + '"', show)
        if len(m.children(n)) <= 2:
            # keep copular verb as a relation
            # in this, you are almost guaranteed to get a tail, a rel, and a head
            pp("Keeping copular verb as a separate relation", show)
            tail = m.words[n]
            rel, head = "", ""
            for c in m.children(n):
                if str(c.deprel) == "cop":
                    rel = c
                    break
            head = find_args(
                m,
                n,
                [
                    ["subj"],
                    ["comp"],
                    ["obj"],
                    ["obl", obl_useful],
                    ["nmod", nmod_useful],
                    ["appos"],
                    ["nummod"],
                ],
                argshow,
            )
            if tail.text and head.text and rel.text:
                rel.state, tail.state = "done", "done"
                rel = rel.text
            else:
                pp("Something is missing among the triplets", show)
                pp(
                    "Head=" + head.text + " Rel=" + rel.text + " Tail=" + tail.text,
                    show,
                )
        else:
            # move copular verb with its parent
            pp("Moving copular verb with its parent", show)
            rel, rel2 = m.words[n].text, ""
            for c in m.children(n):
                if str(c.deprel) == "cop":
                    rel += " " + c.text
                    rel2 = c
                    break
            pp("Relation becomes " + rel, show)
            head = find_args(
                m,
                n,
                [
                    ["subj"],
                    ["comp"],
                    ["obj"],
                    ["obl", obl_useful],
                    ["nmod", nmod_useful],
                    ["appos"],
                    ["nummod"],
                ],
                argshow,
            )
            if (
                str(head.deprel) == "nmod"
                and "done" in m.words[n].state
                and "AUX" not in m.words[n].upos
            ):
                pp(
                    "Head="
                    + head.text
                    + " It is a nmod that is an attribute of "
                    + rel,
                    show,
                )
                sibl = m.siblings(head.id - 1)
                aux_found = "है"
                for sib in sibl:
                    if "AUX" in sib.upos:
                        aux_found = sib.text
                        break
                aux_found = (
                    "property" if is_a_override else aux_found
                )  # <----------- here actual over-riding happens
                tail = head
                head = m.words[n]
                rel = aux_found
            else:
                pp("Head=" + head.text, show)
                tail = find_args(
                    m,
                    n,
                    [
                        ["obj"],
                        ["comp"],
                        ["subj"],
                        ["obl", obl_useful],
                        ["nmod", nmod_useful],
                        ["appos"],
                        ["nummod"],
                    ],
                    argshow,
                )
                if not tail.text:
                    pp("Empty tail found. Let us search in the head only.", show)
                    tail = find_args(
                        m,
                        head.id - 1,
                        [
                            ["obj"],
                            ["subj"],
                            ["obl", obl_useful],
                            ["nmod", nmod_useful],
                            ["appos"],
                            ["nummod"],
                        ],
                        argshow,
                    )  # if the parent cannot give you a tail, then maybe head can give you, check it
                    if tail.text:  # yes, head gave us a tail
                        pp("Tail=" + tail.text, show)
                    else:
                        pp("Still no tail found", show)
                        pp("Let us check for advcl etc etc", show)
                        tail = phrases("", -1, -1, "", "", "", "", "")
                        t1 = find_args(
                            m, n, [["advcl"], ["acl"], ["acl:relcl"]], argshow
                        )
                        if not t1.text:
                            pp("Even now, it is not able to find any tail", show)
                        else:
                            pp('Found an advcl = "' + t1.text + '"', show)
                            t2 = find_args(
                                m,
                                t1.id - 1,
                                [
                                    ["obj"],
                                    ["comp"],
                                    ["subj"],
                                    ["obl", obl_useful],
                                    ["nmod", nmod_useful],
                                    ["appos"],
                                    ["nummod"],
                                ],
                                argshow,
                            )
                            if t2.text:
                                pp('Found an argument "' + t2.text + '" with it', show)
                                tail.text = t2.text + " " + t1.text
                            else:
                                tail = t1
            if tail.text and head.text and rel:
                m.words[n].state = "done"  # rel
                rel2.state = "done"
        if head.text and tail.text:
            extraction_queue.append([head.text, rel, tail.text])
            pp("Extraction obtained are\n", show)
            pp(extraction_queue[-1], show)
        else:
            if head.text:
                head.state = "done"
                pp("This head (" + head.text + ") is done", show)

    elif (
        "advcl" in str(m.words[n].deprel)
        and "mark" not in [str(x.deprel) for x in m.siblings(n)]
        and "mark" not in [str(x.deprel) for x in m.children(n)]
    ):
        # mark represents a breakage of clause i.e. advcl clause becomes somewhat independent
        # the dependent (rel+tail) is the main predicate of the clause
        # the node on which you are standing, can modify the tail or relation, that is why we combine both
        pp('advcl spotted at "' + m.words[n].text + '"', show)
        rel = m.words[n].text
        head = phrases("", -1, -1, "", "", "", "", "")
        tail = find_args(
            m,
            n,
            [
                ["obj"],
                ["comp"],
                ["subj"],
                ["obl", obl_useful],
                ["nmod", nmod_useful],
                ["appos"],
                ["nummod"],
                ["advcl"],
            ],
            argshow,
        )
        parent = m.parent(n)
        ext_found = False
        for ext in extraction_queue:  # checking previous extractions
            if parent.text in ext:  # parent exists in previous extractions,
                if (
                    m.words[n].text not in ext
                ):  # but the advcl node on which you are standing must not exist in that extraction
                    ext_found = ext
                    break
        if ext_found:
            head.text = ext_found[2] + " " + ext_found[1]  # old_predicate + old_rel
        else:
            head = find_args(
                m,
                n,
                [
                    ["subj"],
                    ["comp"],
                    ["obj"],
                    ["obl", obl_useful],
                    ["nmod", nmod_useful],
                    ["appos"],
                    ["nummod"],
                ],
                argshow,
            )

        if head.text and tail.text:
            extraction_queue.append([head.text, rel, tail.text])
            pp("Extraction obtained are\n", show)
            pp(extraction_queue[-1], show)
        else:
            if head.text:
                head.state = "done"
                pp("This head (" + head.text + ") is done", show)

    elif "acl" == str(m.words[n].deprel) or "acl:relcl" == str(m.words[n].deprel):
        # the node on which you are standing modifies the predicate/tail (generally)
        pp('acl spotted at "' + m.words[n].text + '"', show)
        rel = m.words[n].text
        tail = find_args(
            m,
            n,
            [
                ["obj"],
                ["comp"],
                ["subj"],
                ["obl", obl_useful],
                ["nmod", nmod_useful],
                ["appos"],
                ["nummod"],
            ],
            argshow,
        )
        parent = m.parent(n)
        ext_found = False
        for ext in extraction_queue:
            if parent.text in ext and m.words[n].text not in ext:
                ext_found = ext
                break
        if ext_found:
            pp("Previous extraction contains its parent", show)
            head = closest_phrase(
                m, rel, [ext_found[0], ext_found[2]]
            )  # ext_found[0] is old_head, and ext_found[2] is old_tail
        else:
            pp(
                "Parent not found in previous extraction. Let us make parent as head only.",
                show,
            )
            head = parent.text

        if head and tail.text:
            extraction_queue.append([head, rel, tail.text])
            pp("Extraction obtained are\n", show)
            pp(extraction_queue[-1], show)
        else:
            if tail.text:
                tail.state = "done"
                pp("This head (" + tail.text + ") is done", show)

    elif (
        "conj" == str(m.words[n].deprel)
        and m.words[n].head != 0
        and if_any_in(arg_deprels, str(m.parent(n).deprel))
        and m.words[n].state != "conj done"
        and m.words[n].state != "PRON done"
    ):
        pp("conj is seen representing a list", show)
        ext_found = False
        for ext in extraction_queue:
            if m.parent(n).text == ext[0]:
                head = m.words[n].text
                rel = ext[1]
                tail = ext[2]
                ext_found = True
            elif m.parent(n).text == ext[2]:
                head = ext[0]
                rel = ext[1]
                tail = m.words[n].text
                ext_found = True
        m.words[n].state = "conj done"
        if ext_found:
            extraction_queue.append([head, rel, tail])
            pp("Extraction obtained are\n", show)
            pp(extraction_queue[-1], show)
        else:
            pp(
                "Its parent <"
                + m.parent(n).text
                + "> is not present in any previous extractions",
                show,
            )

    else:
        # action verb found
        if is_clausal_node(m, n) and (
            m.words[n].state != "PRON done" and m.words[n].state != "completely done"
        ):  # PRON is pos tag for Pronoun
            # entire clause exists in the node
            pp(m.words[n].text + "<-- This node contains a clause within itself", show)
            head, rel, tail = clausal_node(m, n)
            extraction_queue.append([head, rel, tail])
            pp("here Extraction obtained are", show)
            pp(extraction_queue[-1], show)
            m.words[n].state = "PRON done"
            # m.words[n].upos = m.words[n].upos.replace('PRON','pron')
        else:
            # here we take the phrase (on which you are standing on) as a relation
            pp("I am at node=" + m.words[n].text, show)
            rel = m.words[n].text
            head = find_args(
                m,
                n,
                [
                    ["subj"],
                    ["comp"],
                    ["obj"],
                    ["obl", obl_useful],
                    ["nmod", nmod_useful],
                    ["appos"],
                    ["nummod"],
                ],
                argshow,
            )
            tail = phrases(
                "", -1, -1, "", "", "", "", ""
            )  # <------------------ creating a dummy tail
            if (
                (str(head.deprel) == "nmod" or str(head.deprel) == "appos")
                and "done"
                in m.words[
                    n
                ].state  # nmod merging step i.e. nmod will be treated "is-a" relationship
                and "AUX"
                not in m.words[n].upos  # but there should not be an AUX in its pos tag
                and not (
                    str(m.words[n].deprel) == "nmod"
                    and m.words[n].head != 0
                    and "AUX" in m.parent(n).upos
                )
            ):  # if it itself is an nmod, then there should not be an AUX in its parent (if parent exists)
                pp(
                    "Head="
                    + head.text
                    + " --->  It is a nmod that is an attribute of <"
                    + rel
                    + ">",
                    show,
                )

                # sibl = m.siblings(head.id-1)
                # pp('It has '+str(len(sibl))+' siblings',show)
                def find_nearest_aux(m, n, is_a):
                    # pp('First we search aux in the siblings of <'+rel+'>')
                    return is_a

                aux_found = find_nearest_aux(m, n, is_a)  # is_a #'है'
                # for sib in sibl:
                # 	if 'AUX' in sib.upos:
                # 		aux_found = sib.text
                # 		break
                # 	# else:
                # 	# 	print('*'*100, sib.text)
                aux_found = "property" if is_a_override else aux_found
                tail = head
                head = m.words[n]
                rel = aux_found
            else:
                # so we are sure that it is not a appositive relationship, let us move on now
                pp("Head=" + head.text, show)
                tail = find_args(
                    m,
                    n,
                    [
                        ["obj"],
                        ["comp"],
                        ["subj"],
                        ["obl", obl_useful],
                        ["nmod", nmod_useful],
                        ["appos"],
                        ["nummod"],
                    ],
                    argshow,
                )
                if (
                    head.text
                    and not tail.text
                    and str(m.words[n].deprel) in action_relations
                ):  # we have an head but tail is missing: part 1
                    pp("Empty tail found. Let us search in the head only.", show)
                    tail = find_args(
                        m,
                        head.id - 1,
                        [
                            ["obj"],
                            ["comp"],
                            ["subj"],
                            ["obl", obl_useful],
                            ["nmod", nmod_useful],
                            ["appos"],
                            ["nummod"],
                        ],
                        argshow,
                    )  # agar tail parent ke paas se ni mila, to head ke paas shayad mil jaye
                    if not tail.text:
                        pp("Still no tail found", show)
                if (
                    head.text and not tail.text
                ):  # # we have an head but tail is missing: part 2
                    if is_clausal_node(
                        m, m.words[n].head - 1
                    ):  # check if its parent is a clausal node or not
                        pp(
                            "Since the parent of <"
                            + m.words[n].text
                            + "> is a clausal node",
                            show,
                        )
                        pp("\t let us pick its relation and tail", show)
                        a, b, c = clausal_node(m, m.words[n].head - 1)
                        tail = head
                        head = phrases(
                            c, -1, -1, "", "", "", "", ""
                        )  # <---------notice this is not a dummy, head,text will be c
                        rel = b
                    elif m.words[n].head != 0:
                        pp(
                            "Since parent of <"
                            + m.words[n].text
                            + "> exists, let us make it tail only",
                            show,
                        )
                        tail = m.parent(n)
                    else:
                        pp("Looking into previous extractions", show)
                        parent = m.words[n]
                        ext_found = False
                        for ext in extraction_queue:
                            if parent.text in ext and head.text not in ext:
                                ext_found = ext
                                # break
                        if ext_found:
                            pp("Found in previous extractions", show)
                            tail.text = closest_phrase(
                                m, head.text, [ext_found[0], ext_found[2]]
                            )
                        else:
                            pp("Still no tail found", show)
                            pp("Let us check for advcl etc etc", show)
                            tail = phrases(
                                "", -1, -1, "", "", "", "", ""
                            )  # <------------------ creating a dummy tail
                            t1 = find_args(
                                m, n, [["advcl"], ["acl"], ["acl:relcl"]], argshow
                            )
                            if not t1.text:
                                pp("Even now, it is not able to find any tail", show)
                            else:
                                pp('Found an advcl = "' + t1.text + '"', show)
                                t2 = find_args(
                                    m,
                                    t1.id - 1,
                                    [
                                        ["obj"],
                                        ["comp"],
                                        ["subj"],
                                        ["obl", obl_useful],
                                        ["nmod", nmod_useful],
                                        ["appos"],
                                        ["nummod"],
                                    ],
                                    argshow,
                                )
                                if t2.text:
                                    pp(
                                        'Found an argument "' + t2.text + '" with it',
                                        show,
                                    )
                                    tail.text = t2.text + " " + t1.text
                                else:
                                    tail = t1
                if tail.text:
                    pp("Tail=" + tail.text, show)
            if head.text and tail.text:
                extraction_queue.append([head.text, rel, tail.text])
                pp("Extraction is", show)
                pp(extraction_queue[-1], show)
                m.words[n].state = "done" if not m.words[n].state else m.words[n].state
            else:
                head.state = "done" if head.text != "" else ""
            # print(head.text)
            # input('wait')

    # input('this')
    if running_no >= 300:
        raise "Infinite loop it seems"

    pp(
        "Checking for leftovers...", show
    )  # by leftovers, here we mean those nodes, that can be a valid or a tail
    leftover = find_args(
        m,
        n,
        [
            ["subj"],
            ["comp"],
            ["obj"],
            ["obl", obl_useful],
            ["nmod", nmod_useful],
            ["appos"],
            ["nummod"],
        ],
        argshow,
        stateless=True,
    )
    if leftover.text:  #'done' not in c.state and if_any_in(arg_deprels,c.deprel):
        pp(
            'There are still some potential entities left in "' + m.words[n].text + '"',
            show,
        )
        pp('\tLike "' + leftover.text + '"', show)
        extraction_queue = extract(
            m, n, extraction_queue, is_a, running_no + 1, show, argshow, is_a_override
        )  # <-------- calling recursive
    m.words[n].state = "completely done"
    pp('No leftovers found for "' + m.words[n].text + '"', show)

    for c in m.children(n):  # let us move to its children now
        if m.children(c.id - 1) or "conj" == str(
            c.deprel
        ):  # if that child is not a leaf node or if that child has a conj deprel
            pp("Looking at the child <" + c.text + "> now", show)
            extraction_queue = extract(
                m,
                c.id - 1,
                extraction_queue,
                is_a,
                running_no + 1,
                show,
                argshow,
                is_a_override,
            )  # <-------- calling recursive
        else:
            c.state = "leaf done"
    return extraction_queue


def load_stanza_model(lang, use_gpu=False):
    # 1 = will not download anything, probably resulting in failure if the resources aren't already in place.
    # 2 = will reuse the existing resources.json and models, but will download any missing models.
    # 3 = will download a new resources.json and will overwrite any out of date models.
    try:
        # nlp = stanza.Pipeline(lang,dir=stanza_path+'/'+stanza_version+'/',download_method=2,use_gpu=use_gpu) #stanza.Pipeline(lang, use_gpu=use_gpu)
        nlp = stanza.Pipeline(
            lang, use_gpu=use_gpu
        )  # stanza.Pipeline(lang, use_gpu=use_gpu)
    except:
        stanza.download(
            lang, dir=stanza_path + "/" + stanza_version + "/"
        )  # or model_dir=stanza_path+'/'+stanza_version+'/'
        nlp = stanza.Pipeline(
            lang,
            dir=stanza_path + "/" + stanza_version + "/",
            download_method=2,
            use_gpu=use_gpu,
        )  # stanza.Pipeline(lang, use_gpu=use_gpu)
    return nlp
