# Mini GPT-2
A replication of GPT-2 on a smaller scale, utilizing fewer parameters while trying to replicate similar outcomes.
## What This Is
_A more in depth description about the mini-gpt project and more about some brief design decisions_
## Architecture
Word Embeddings -> [X + MultiHeadAttention -> LayerNorm -> X_out + MultiLayerPercerptron -> LayerNorm] * K -> LayerNorm -> Linear -> CrossEntropyLoss

## Results
### Metrics

### Model Performance

## Design Decisions
Across the project, several decisions had to be made, such as the model size, with the addition of the decision of the number of layers, number of heads, and dimensions of the model. The decisions also stretched towards the data to be utilized in the model and what specifically. Furthermore, there was a decided amount of 30 epochs initially.

### Choosing the Data
There are several typical means of gathering data to train a model, with the most common being Subreddits, Wikipedia, and online scraping. The primary decision was to realize that the best source for training the data was Wikipedia, as it is a free-use platform and has an API that can be used to pull articles. The next procedure was to choose what data would be best to train the Mini GPT-2 on. The primary choice was to utilize the Formula 1 drivers, but as stated later, more data was needed to allow for better validation loss of the model, as while the training loss was becoming gradually better, the validation loss was not. The change made was to add Formula 2 drivers, the circuits F1 and F2 race on, and past seasons.

The main reason for choosing the Formula racing series, their tracks, and history is due to the fact that I am quite knowledgeable about them, making it easier for me to validate what the model presents.

### Initial Design Decisions for Model Parameters
The standard GPT-3 model maintains 96 layers of the transformer block and a proportionate number of heads between the word embeddings and the layer normalization prior. But this isn't something that would fit the model, and usually requires a large amount of tokens and unique vocab, which is not at the expense of the model, and in addition goes against the fact that we aim to build a Mini GPT-2 model. So to initialize the project, it began with 10 layers, 8 heads, and 256 dimensions for the model. To emphasize the number of heads, it must proportionately divide the dimensions of the model, and it is typically better to have a small amount.

### Final Design Decisions for Model Parameters
The table below contains tracked changes that were made across training, each one making progress in reducing the validation loss. It is also quite visible that the loss takes a turn around epoch 6, leading to the decision to reduce the epochs further down to this point to prevent factors like overfitting and overtraining. Another key point to make was the reduction of the number of layers as well as model dimensions. This was done as in some cases the training loss went down to 0.02 very fast, indicating the fact that there was overfitting present in the model and that it was tending to memorize rather than learn.

| Change | Best val loss | Turning point |
| --- | --- | --- |
| Baseline (22 articles, 10 layers, d=256, 8.5K vocab) | 5.04 | epoch 6 |
| More data + smaller model (5 layers, d=128) |	4.75 | epoch 7 |
| Min-frequency vocab cutoff (15.3K → ~6.3K) |	4.32 | epoch 6 |

## How to Run the Model

## What I Learned

## References
- Understanding Deep Learning. _Understanding Deep Learning_ [https://udlbook.github.io/udlbook/](url)
