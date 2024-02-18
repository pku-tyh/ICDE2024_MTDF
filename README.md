# Multi-View Teacher with Curriculum Data Fusion for Robust Unsupervised Domain Adaptation

## Abstract
Graph Neural Networks (GNNs) have emerged as an effective tool for graph classification, yet their reliance on extensive labeled data poses a significant challenge, especially when such labels are scarce. To address this challenge, this paper presents a novel framework, denoted as Multi-View Teacher with Curriculum Data Fusion (MTDF). MTDF achieves robust unsupervised domain adaptation in both the model and data perspectives. On the one hand, MTDF utilizes a multi-teacher framework with diverse update strategies for robust adaptation. Moreover, it employs a complementary perspective consistency model from local implicit representation and global explicit graph structure. On the other hand, MTDF generates source-mimicry data at the target domain to serve as a bridge to overcome the challenge of domain shift. 
MTDF achieve stable unsupervised domain adaptation through bi-directional processes from the perspective of both the model and the data.
We have conducted comprehensive experimental evaluations across multiple real-world datasets with a range of baseline methods to demonstrate the superior performance of our proposed method.

## Code Usage

1. Papare a environment with Python 3.8 and Cuda 11.1

2. Install requirments

```shell
pip install -r requirments.txt
```

3. Run training & testing script

```shell
bash run.sh
```

