# Mini GPT-2
A replication of GPT-2 on a smaller scale, utilizing fewer parameters while trying to replicate similar outcomes.
## What This Is
_A more in depth description about the mini-gpt project and more about some brief design decisions_
## Architecture
Word Embeddings -> [X + MultiHeadAttention -> LayerNorm -> X_out + MultiLayerPercerptron -> LayerNorm] * K -> LayerNorm -> Linear -> CrossEntropyLoss

**Parameters for Model**
- num_layers = 5
- num_heads = 8
- d_model = 128
- vocab_size = 4,352 (length of the vocab from the data loaded)
- seq_len = 128

## Results
### Metrics
| Change | Best val loss (ln(vocab_size)) | Turning point |
| --- | --- | --- |
| Baseline (22 articles, 10 layers, d=256, 8.5K vocab) | 5.04 | epoch 6 |
| More data + smaller model (5 layers, d=128) |	4.75 | epoch 7 |
| Min-frequency vocab cutoff (15.3K → ~6.3K) |	4.33 | epoch 6 |

### Model Performance
#### Poor Performance
```
Enter a prompt: Lewis Hamilton
Lewis Hamilton in the first lap of the season . Hamilton took pole at the Hungarian Grand Prix , but was promoted Hamilton to Hamilton to Hamilton to Vettel , but Hamilton to Hamilton to Hamilton the lead , while Hamilton , but the lead the lead of the race from the race from Hamilton , and Hamilton . Hamilton was able to start , but the lead from the race , but lost the lead to pass Hamilton , while Räikkönen pitted on the lead . Hamilton was able to take his first lap , but was able to Hamilton '
```
Earlier, before making several changes outlined in the **Debugging Narrative** and the **Final Design Decision**, the model was tending to repeat words quite often and was only operating on a regular softmax -> multimodal structure for decoding, lacking temperature and top-k, meaning that some tokens were chosen in completely random places.
#### Good Performance
```
Enter a prompt: Lewis Hamilton won
Lewis Hamilton won the championship title race at the 2005 season. Red Bull 's nine race victory win the race of the season, Hamilton was the championship by two rounds, tied to Michael Schumacher 's record number also becoming securing a record in most race of consecutive wins including a win at a overall pace which soon had a second win in the longest him recognised by Nissan fourth place. The 2010 performance also a Thai flag of June season to fans in speed of the Red Bull junior. After the sport and Sauber Marussia drivers
```
The model can form coherent sentences, but with some errors, and falls short on connecting the context of these sentences. However, that is due to constraints with the model size, and also the small amount of training data.
## Design Decisions
Across the project, several decisions had to be made, such as the model size, with the addition of the decision of the number of layers, number of heads, and dimensions of the model. The decisions also stretched towards the data to be utilized in the model and what specifically. Furthermore, there was a decided amount of 30 epochs initially.

### Choosing the Data
There are several typical means of gathering data to train a model, with the most common being Subreddits, Wikipedia, and online scraping. The primary decision was to realize that the best source for training the data was Wikipedia, as it is a free-use platform and has an API that can be used to pull articles. The next procedure was to choose what data would be best to train the Mini GPT-2 on. The primary choice was to utilize the Formula 1 drivers, but as stated later, more data was needed to allow for better validation loss of the model, as while the training loss was becoming gradually better, the validation loss was not. The change made was to add Formula 2 drivers, the circuits F1 and F2 race on, and past seasons.

The main reason for choosing the Formula racing series, their tracks, and history is due to the fact that I am quite knowledgeable about them, making it easier for me to validate what the model presents.

### Initial Design Decisions for Model Parameters
The standard GPT-3 model maintains 96 layers of the transformer block and a proportionate number of heads between the word embeddings and the layer normalization prior. But this isn't something that would fit the model, and usually requires a large amount of tokens and unique vocab, which is not at the expense of the model, and in addition goes against the fact that we aim to build a Mini GPT-2 model. So to initialize the project, it began with 10 layers, 8 heads, and 256 dimensions for the model. To emphasize the number of heads, it must proportionately divide the dimensions of the model, and it is typically better to have a small amount.

### Final Design Decisions for Model Parameters
The table below contains tracked changes that were made across training, each one making progress in reducing the validation loss. It is also quite visible that the loss takes a turn around epoch 6, leading to the decision to reduce the epochs further down to this point to prevent factors like overfitting and overtraining. Another key point to make was the reduction of the number of layers as well as model dimensions. This was done as in some cases the training loss went down to 0.02 very fast, indicating the fact that there was overfitting present in the model and that it was tending to memorize rather than learn.

| Change | Best val loss (ln(vocab_size)) | Turning point |
| --- | --- | --- |
| Baseline (22 articles, 10 layers, d=256, 8.5K vocab) | 5.04 | epoch 6 |
| More data + smaller model (5 layers, d=128) |	4.75 | epoch 7 |
| Min-frequency vocab cutoff (15.3K → ~6.3K) |	4.33 | epoch 6 |

### Debugging Narrative 

## How to Run the Model
To install libraries:
```
pip install -r requirements.txt
```

How to run `fetch_f1_wiki.py`:
```
python -m data.fetch_f1_wiki
```

How to run `train.py`:
```
python -m src.train
```

How to run `generate.py`:
```
python -m src.generate
```

## What I Learned
At first, there were a few issues that I struggled with while creating the model that led to major learning points. These include instances such as memorization/overfitting diagnosis, shape issues in attention, the softmax being applied twice, and the tokenizer edge case.
### Memorization/Overfitting
Initially, during training, it achieved a training loss of 0.003, leading to the assumption that it was performing very well. When looking into ways to generate the text, I realized that the model was actually memorizing; when prompted, it tended to perform poorly in practice. Later on, I had to realize that the lack of a validation and test set led to a gap in understanding, as those would truly allow us to realize the model's performance.

After diagnosing that problem, this led to a further deep dive into other underlying issues causing this memorization, such as a small dataset with a small vocabulary (22 driver articles) on a big model (10-layer, 256-dimensional model) and non-overlapping windows leading to fewer gradient updates per epoch. 

A small data set on such a large model will cause the self-attention to keep being fed into itself continuously until the point that it is almost a matrix of 1s, meaning that the model in turn will end up always producing a near 0.0 loss, which led to me realizing that with a small data set, it is better to reduce the dimensionality of the model and also the number of transformer blocks to combat this issue.

The next realization is that the lack of overlapping windows led to very few gradient updates per epoch. With a stride equal to the sequence length, few windows were produced, and the model saw a very small fixed set of examples again and again, leading to this sort of memorization.

### Attentions Shape Issues

### Softmax-Applied-Twice

## References
- Understanding Deep Learning. _Understanding Deep Learning_ [https://udlbook.github.io/udlbook/](url)
- Speech and Language Processing. _Words and Tokens_ [https://web.stanford.edu/~jurafsky/slp3/2.pdf](url)
