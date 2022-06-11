

from transformers import AutoModel, AutoTokenizer

model = AutoModel.from_pretrained("distilbert-base-uncased")
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

# get the current vocabulary
vocabulary = tokenizer.get_vocab().keys()

new_words = ["new_word1", "new_word2"]
for word in new_words:
    # check to see if new word is in the vocabulary or not
    if word not in vocabulary:
        tokenizer.add_tokens(word)

# add new embeddings to the embedding matrix of the transformer model
model.resize_token_embeddings(len(tokenizer))
