# RecBole-Debias

**RecBole-Debias** is a toolkit built upon [RecBole](https://github.com/RUCAIBox/RecBole) for reproducing and developing debiased recommendation algorithms.

## Highlights

* **Unified**

    Unified framework, which includes several algorithms for different kinds of biases. Meanwhile, three datasets in which the distribution of training set and test set is different are provided for evaluation.

* **Adaptive**

    Adaptive to many base recommendation models. For simplicity, the current implementation is only based on MF model.
    
* **Closely**

    Closely related to Recbole. The toolkit fully adopts the functions of Recbole, except that certain algorithms need to design unique components like trainer.

## Requirements

```
python>=3.7.0
pytorch>=1.7.0
recbole>=1.0.1
```

## Quick-Start

With the source code, you can use the provided script for initial usage of our library:

```bash
python run_recbole_debias.py
```

If you want to change the models or datasets, just run the script by setting additional command parameters:

```bash
python run_recbole_debias.py -m [model] -d [dataset] -c [config_files]
```

## Implemented Models

We list currently supported models according to category:

**Base Model**:

* **[MF](recbole_debias/model/debiased_recommender/mf.py)** from Koren *et al.*: [Matrix factorization techniques for recommender systems](https://www.inf.unibz.it/~ricci/ISR/papers/ieeecomputer.pdf) (Computer 2009).

**Selection Bias**:

* **[MF-IPS](recbole_debias/model/debiased_recommender/mf_ips.py)** from Schnabel *et al.*: [Recommendations as Treatments: Debiasing Learning and Evaluation](http://proceedings.mlr.press/v48/schnabel16.pdf) (ICML 2016).

**Popularity Bias**:

* **[PDA](recbole_debias/model/debiased_recommender/pda.py)** from Zhang *et al.*: [Causal intervention for leveraging popularity bias in recommendation
](https://arxiv.org/pdf/2105.06067.pdf) (SIGIR 2021).
* **[MACR](recbole_debias/model/debiased_recommender/macr.py)** from Wei *et al.*: [Model-Agnostic Counterfactual Reasoning for Eliminating Popularity Bias in Recommender System](https://arxiv.org/pdf/2010.15363.pdf) (KDD 2021).
* **[DICE](recbole_debias/model/debiased_recommender/dice.py)** from Zheng *et al.*: [Disentangling User Interest and Conformity for Recommendation with Causal Embedding](https://arxiv.org/pdf/2006.11011.pdf?ref=https://githubhelp.com) (WWW 2021).
* **[CausE](recbole_debias/model/debiased_recommender/cause.py)** from Bonner *et al.*: [Causal Embeddings for Recommendation](https://arxiv.org/pdf/1706.07639.pdf?ref=https://githubhelp.com) (RecSys 2018).

**Exposure Bias**:

* **[Rel-MF](recbole_debias/model/debiased_recommender/rel_mf.py)** from Yuta *et al.*: [Unbiased Recommender Learning from Missing-Not-At-Random Implicit Feedback](https://arxiv.org/pdf/1909.03601.pdf) (WSDM 2020).

## Datasets
 The datasets used can be downloaded from [Datasets Link](https://drive.google.com/drive/folders/1W6fvJN9ZjuyeqsIuUeodDJk_ajajHkoG).

## Details

- Details on [`Yahoo!R3`](details/yahoo.md) dataset; 
- Details on [`KuaiRec`](details/kuai.md) dataset; 
- Details on [`MovieLens-100k`](details/ml.md) dataset; 

## The Team

RecBole-Debias is developed and maintained by members from [RUCAIBox](http://aibox.ruc.edu.cn/), the main developers is Jingsen Zhang ([@Jingsen Zhang](https://github.com/JingsenZhang)).

## Acknowledgement

The implementation is based on the open-source recommendation library [RecBole](https://github.com/RUCAIBox/RecBole).

Please cite the following paper as the reference if you use our code or processed datasets.

```
@inproceedings{zhao2021recbole,
  title={Recbole: Towards a unified, comprehensive and efficient framework for recommendation algorithms},
  author={Wayne Xin Zhao and Shanlei Mu and Yupeng Hou and Zihan Lin and Kaiyuan Li and Yushuo Chen and Yujie Lu and Hui Wang and Changxin Tian and Xingyu Pan and Yingqian Min and Zhichao Feng and Xinyan Fan and Xu Chen and Pengfei Wang and Wendi Ji and Yaliang Li and Xiaoling Wang and Ji-Rong Wen},
  booktitle={{CIKM}},
  year={2021}
}
@inproceedings{recbole[2.0],
  title={RecBole 2.0: Towards a More Up-to-Date Recommendation Library},
  author={Zhao, Wayne Xin and Hou, Yupeng and Pan, Xingyu and Yang, Chen and Zhang, Zeyu and Lin, Zihan and Zhang, Jingsen and Bian, Shuqing and Tang, Jiakai and Sun, Wenqi and others},
  booktitle={Proceedings of the 31st ACM International Conference on Information \& Knowledge Management},
  pages={4722--4726},
  year={2022}
}

