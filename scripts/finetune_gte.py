# %%
import pandas as pd
import nltk
nltk.download('punkt', quiet=False)
nltk.download('punkt_tab')

from sentence_transformers import SentenceTransformer, models, InputExample, losses

import re 

import sys
sys.path.append('../src')

from preprocessing import *
from models import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# %%
dfs = get_dfs(os.path.dirname(os.getcwd()))
static_df = create_static_df(dfs)
notes = create_notes_df(dfs, filename=None)

# %%
# creates [num] sentences from clinical notes and saves them to data/finetuning/[saveas].txt
num = 10_000
saveas = 'sentences'

def save_sentences_to_txt(sentences, filename):
    with open(filename, 'w', encoding='utf-8') as file:
        for sentence in sentences:
            file.write(sentence + '\n')

sentences = []

# Shuffle the texts and sample num_sentences many which should be enough to produce num_sentences
print('Processing notes...')
data = notes.copy()

# Shuffle the texts and sample the first 'num' rows
data = data.sample(frac=1, random_state=43).reset_index(drop=True)
data = data.head(num)
all_texts = data["text"].tolist()

for text in all_texts:
    text = text.replace('___', '') # remove deidentifiers
    text = text.replace('\n', ' ')
    pattern = r"\s+"  # Matches one or more whitespace characters
    text = re.sub(pattern, " ", text)  # reduce multiple whitespaces to a single one
    tok_sentences = nltk.sent_tokenize(text)
    tok_sentences = [s for s in tok_sentences if len(s.split()) >= 5]  # require at least 5 words to avoid fragments
    tok_sentences = [s for s in tok_sentences if len(s.split()) < 500]  # require at most 500 words due to context limit of model, most sentences will be shorter anyway
    sentences.extend(tok_sentences)
    sentences = sentences[:num]

save_sentences_to_txt(sentences, f"../data/finetuning/{saveas}.txt")
print(f"Saved {len(sentences)} to ../data/finetuning/{saveas}.txt")

# %%
def freeze_first_n_layers(model, n):
    # freezes first n encoder layers 
    # gte-large has 24 encoder blocks indexed 0 up to 23
    for name, param in model.named_parameters():
            if 'encoder.layer.' in name:
                layer_num = int(name.split('.')[4])
                if layer_num < n:
                    param.requires_grad = False

# %%
# Ensure to log in first: huggingface-cli login

model_name="thenlper/gte-large"
#model_name = "sn2727/med-gte-simcse"
model = SentenceTransformer(model_name).to(device)

# optionally freeze layers
freeze_first_n_layers(model, 21)

# %%
# Finetuning 

# first load sentences
SENTENCES_FILE = "../data/finetuning/sentences10k.txt"
sentences = []
        
with open(SENTENCES_FILE, 'r', encoding='utf-8') as file:
    # Read given file
    sentences = [sentence.strip() for sentence in file.readlines()]

print(f'Loaded {len(sentences)} sentences')

# Convert train sentences to sentence pairs
data = [InputExample(texts=[s, s]) for s in sentences]
dataloader = DataLoader(data, batch_size=32, shuffle=True)

# Use the denoising auto-encoder loss
train_loss = losses.MultipleNegativesRankingLoss(model)

model.fit(
    train_objectives=[(dataloader, train_loss)],
    epochs=1,
    optimizer_params={'lr': 5e-7},
    show_progress_bar=True,
)

model.save("../models/med-gte-simcse-ger")



