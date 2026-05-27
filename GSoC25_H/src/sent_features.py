import torch


def find_mentions(sentence, ner_tags, pos_tags):
    words = sentence
    mentions = []
    current_mention = []

    for word, ner, pos in zip(words, ner_tags, pos_tags):
        if ner.startswith("B-"):  # Beginning of a new entity
            if current_mention:
                mentions.append(" ".join(current_mention))
                current_mention = []
            current_mention.append(word)
        elif ner.startswith("I-"):  # Inside an entity
            current_mention.append(word)
        else:  # Outside an entity
            if current_mention:
                mentions.append(" ".join(current_mention))
                current_mention = []

            # Check for noun phrases that might be mentions
            if pos.startswith("N"):  # Noun
                mentions.append(word)

    # Add the last mention if there's any
    if current_mention:
        mentions.append(" ".join(current_mention))

    return mentions


def get_ner_tags(sentence, tokenizer, model):
    tok_sentence = tokenizer(
        sentence, return_tensors="pt", padding=True, truncation=True, max_length=512
    )

    with torch.no_grad():
        logits = model(**tok_sentence).logits.argmax(-1)
        predicted_tokens_classes = [model.config.id2label[t.item()] for t in logits[0]]
        predicted_labels = []
        previous_token_id = 0
        word_ids = tok_sentence.word_ids()
        for word_index in range(len(word_ids)):
            if word_ids[word_index] == None:
                previous_token_id = word_ids[word_index]
            elif word_ids[word_index] == previous_token_id:
                previous_token_id = word_ids[word_index]
            else:
                predicted_labels.append(predicted_tokens_classes[word_index])
                previous_token_id = word_ids[word_index]
        return predicted_labels


def get_pos_ner_tags(text, nlp, tokenizer, model):
    doc = nlp(text)
    pos_ner_tags = []
    mentions = []
    for sentence in doc.sentences:
        words = [word.text for word in sentence.words]
        ner_tags = get_ner_tags(" ".join(words), tokenizer, model)
        pos_tags = []
        for idx, word in enumerate(sentence.words):
            pos_ner_tags.append((word.text, word.pos, ner_tags[idx]))
            pos_tags.append(word.pos)
        mentions.append(find_mentions(words, ner_tags, pos_tags))
    mentions = [mention for sublist in mentions for mention in sublist]
    mentions = list(set(mentions))
    return pos_ner_tags, mentions
