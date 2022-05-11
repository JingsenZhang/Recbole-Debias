# Experimental Setting

**Dataset:** Yahoo!R3

**Processing:** 

Split the dataset (consisting of normal set and intervened set) as follows:
- training set (100% normal and 25% intervened)
- validation set (25% intervened)
- test set (50% intervened)

**Filtering:** 
- Remove users and items with less than 1 interactions

**Evaluation:** Ratio-based Splitting, full sort

**Metrics:** Recall@10, MRR@10, NGCG@10, Hit@10, Precision@10

**Properties:**

```yaml
# Atomic File Format
field_separator: "\t"
seq_separator: " "

# Common Features
USER_ID_FIELD: user_id
ITEM_ID_FIELD: item_id
RATING_FIELD: rating
TIME_FIELD: timestamp
seq_len: ~
# Label for Point-wise DataLoader
LABEL_FIELD: label
# NegSample Prefix for Pair-wise DataLoader
NEG_PREFIX: neg_
# Sequential Model Needed
ITEM_LIST_LENGTH_FIELD: item_length
LIST_SUFFIX: _list
MAX_ITEM_LIST_LENGTH: 50
POSITION_FIELD: position_id
# Knowledge-based Model Needed
HEAD_ENTITY_ID_FIELD: head_id
TAIL_ENTITY_ID_FIELD: tail_id
RELATION_ID_FIELD: relation_id
ENTITY_ID_FIELD: entity_id

# Selectively Loading
load_col:
    inter: [user_id, item_id, rating, intervene_mask]
unload_col: ~
unused_col: ~

# Filtering
rm_dup_inter: ~
val_interval: ~
filter_inter_by_user_or_item: True
user_inter_num_interval: '[1,inf)'
item_inter_num_interval: '[1,inf)'

# Benchmark file
benchmark_filename: ['train','valid','test']

# special
INTERVENE_MASK: intervene_mask
```

# Dataset Statistics

| Dataset    | #Users | #Items | #Interactions | Sparsity |
| ---------- | ------ | ------ | ------------- | -------- |
| Yahoo!R3   | 15,400 | 1,000  |   365,704     | 97.62%   |

# Evaluation Results

| Method               | Recall@10 | MRR@10 | NDCG@10 | Hit@10 | Precision@10 |
| -------------------- | --------- | ------ | ------- | ------ | ------------ |
| **MF**               | 0.0151    | 0.0227 | 0.0119  | 0.0741 | 0.0076       |
| **MF-IPS (User)**    | 0.0147    | 0.0218 | 0.0115  | 0.0698 | 0.0073       |
| **MF-IPS (Item)**    | 0.0149    | 0.0235 | 0.0120  | 0.0730 | 0.0074       |
| **MF-IPS (NB)**      | 0.0149    | 0.0237 | 0.0121  | 0.0730 | 0.0074       |
| **PDA**              | 0.0166    | 0.0304 | 0.0145  | 0.0807 | 0.0083       |
| **MACR**             | 0.0164    | 0.0280 | 0.0138  | 0.0792 | 0.0082       |
| **DICE**             | 0.0166    | 0.0318 | 0.0149  | 0.0802 | 0.0083       |
| **CausE**            | 0.0123    | 0.0192 | 0.0098  | 0.0611 | 0.0061       |
| **URL**              | 0.0164    | 0.0279 | 0.0137  | 0.0806 | 0.0082       |

# Hyper-parameters
For fairness, we tune the common hyper-parameters of methods as following. 
```
learning_rate in [0.01, 0.005, 0.001, 0.0001]
embedding_size in [16, 32, 64]
train_batch_size in [256, 512, 1024, 2048]
```

|                      | Best hyper-parameters                                        | Tuning range                                                 |
| -------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| **MF**               | learning_rate=0.005<br />embedding_size=64<br />train_batch_size=2048 |-  |
| **MF-IPS (User)**    | learning_rate=0.001<br />embedding_size=64<br />train_batch_size=256<br />eta=0.6 | eta in [0.1, 0.2, ..., 1] |
| **MF-IPS (Item)**    | learning_rate=0.001<br />embedding_size=16<br />train_batch_size=1024 |-  |
| **MF-IPS (NB)**      | learning_rate=0.005<br />embedding_size=64<br />train_batch_size=2048 |-  |
| **PDA**              | learning_rate=0.001<br />embedding_size=64<br />train_batch_size=2048<br />reg_weight=0.001 | reg_weight in [0.1, 0.01, 0.001] |
| **MACR**             | learning_rate=0.001<br />embedding_size=64<br />train_batch_size=512<br />item_loss_weight=0.01<br />user_loss_weight=0.001<br />mlp_hidden_size=[32, 1]<br />dropout_prob=0.1<br />c=5 | item_loss_weight in [0.001, 0.005, 0.01, 0.05, 0.1]<br />user_loss_weight in [0.001, 0.005, 0.01, 0.05, 0.1]<br />mlp_hidden_size in [[32, 1], [32, 16, 1]]<br />dropout_prob in [0.1, 0]<br />c in [5, 1, 0.1, 0] |
| **DICE**             | learning_rate=0.001<br />embedding_size=64<br />train_batch_size=2048<br />dis_pen=0.01<br />int_weight=0.1<br />pop_weight=0.0001 | dis_pen in [0.1, 0.01, 0.001]<br />int_weight in [0.1, 0.5, 0.01, 0.05, 0.0001]<br />pop_weight in [0.1, 0.5, 0.01, 0.05, 0.0001]|
| **CausE**            | learning_rate=0.005<br />embedding_size=32<br />train_batch_size=256<br />dis_pen=0.1 | dis_pen in [0.1, 0.01, 0.001] |
| **URL**              | learning_rate=0.001<br />embedding_size=32<br />train_batch_size=2048<br />reg_weight=0.001<br />eta=0.5 | reg_weight in [0.1, 0.01, 0.001]<br />eta in [1, 0.8, 0.7, 0.5] |
