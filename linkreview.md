# LinkReview

- Here we have collect info about all the works that may be useful for writing our paper

> [!NOTE]
> This review table will be updated, so it is not a final version

| Title | Year | Authors | Paper | Summary |
| :---: | :---: | :---: | :---: | :---: |
|  Neural oblivious decision ensembles for deep learning on tabular data |  2019  | Popov, Morozov, Babenko | [arxiv](https://arxiv.org/pdf/1909.06312) | First paper from Yandex Research about Tabular DL. The authors present a neural network based on smoothed decision trees. In the following studies it is shown that they do not perform well |
|  Revisiting deep learning models for tabular data |  2021  | Gorishniy, Rubachev, Khrulkov, Babenko | [NeurIPS](https://proceedings.neurips.cc/paper_files/paper/2021/file/9d86d83f925f2149e9edb0ac3b49229c-Paper.pdf) | A great overview of methods for Tabular DL, also the authors present two neural network architectures for tabular data: ResNet and FT-Transformer. Transformer performs better, however it takes a long time to learn it |
|  On embeddings for numerical features in tabular deep learning |  2022  | Gorishniy, Rubachev, Babenko | [NeurIPS](https://proceedings.neurips.cc/paper_files/paper/2022/file/9e9f0ffc3d836836ca96cbf8fe14b105-Paper-Conference.pdf) | Embedings for TabularDL: piecewise linear encoding (generelization of one-hot encoding) + periodic activations (sin, cos) |
|  TabR: Tabular Deep Learning Meets Nearest Neighbors |  2023  | Gorishniy, Rubachev, Kartashev, Shlenskii, Kotelnikov, Babenko | [arxiv](https://arxiv.org/pdf/2307.14338) <br> [GitHub](https://github.com/yandex-research/tabular-dl-tabr) | Last paper from Yandex Research about Tabular DL. The authors propose a kNN-based neural network, but the similarity search methods and object selection are more complex |
|SAINT: Improved neural networks for tabular data via row attention and contrastive pre-training  |  2021  | Somepalli, Goldblum, Schwarzschild, Bruss, Goldstein | [arxiv](https://arxiv.org/pdf/2106.01342) | Retrieval-augmented models for tabular data problems. <br> More papers about it in the [TabR](https://arxiv.org/pdf/2307.14338) paper |
|      |      |      |      |      |
| Prodigy: An expeditiously adaptive parameter-free learner | 2024 | Mishchenko, Defazio | [GitHub](https://github.com/konstmish/prodigy) <br> [arxiv](https://arxiv.org/pdf/2306.06101)| Parameter-free optimization method for DL |
| Learning-Rate-Free Learning by D-Adaptation | 2023 | Defazio, Mishchenko | [GitHub](https://github.com/facebookresearch/dadaptation) <br> [arxiv](https://arxiv.org/pdf/2301.07733)| Another parameter-free optimization method for DL |
| Symbolic Discovery of Optimization Algorithms (LION) | 2023 | Chen et al. | [arxiv](https://arxiv.org/pdf/2301.07733) <br> [GitHub](https://github.com/google/automl/tree/master/lion)| Otimizator LION that is better than Adam (according to authors) |
| signSGD: Compressed Optimization for Non-Convex Problems | 2018 | Bernstein et al. | [arxiv](https://arxiv.org/pdf/1802.04434) <br> [GitHub](https://github.com/jxbz/signSGD/blob/master/signSGD_zeros.ipynb)| signSGD, should work well in the fine-tuning, maybe will work well in TabularDL |
|      |      |      |      |      |
| Robust optimization for adversarial learning with finite sample complexity guarantees | 2024 | Bertolace et al. | [arxiv](https://arxiv.org/pdf/2403.15207)| Adversarial training |
| Understanding adversarial training: Increasing local stability of supervised models through robust optimization | 2018 | Shaham et al. | [Neurocomputing](https://www.sciencedirect.com/science/article/pii/S0925231218304557)| Adversarial training |
| Interpreting Robust Optimization via Adversarial Influence Functions | 2020 | Deng et al. | [ICML](http://proceedings.mlr.press/v119/deng20a/deng20a.pdf)| Adversarial training |
| Adversarial Distributional Training for Robust Deep Learning | 2020 | Dong et al. | [NeurIPS](https://proceedings.neurips.cc/paper_files/paper/2020/file/5de8a36008b04a6167761fa19b61aa6c-Paper.pdf)| Adversarial training |
