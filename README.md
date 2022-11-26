# IGRP: Iterative Gradient Rank Pruning for Finding Graph Lottery Ticket
[IEEE BigData 2022] IGRP: Iterative Gradient Rank Pruning for Finding Graph Lottery Ticket

Po-wei Harn, Sai Deepthi Yeddula, Bo Hui, Jie Zhang, Libo Sun, Min-Te Sun, Wei-Shinn Ku

## Overview
Graph Neural Networks (GNNs) have shown promising
performance in many applications, yet remain extremely difficult to train over large-scale graph datasets. 
Existing weight pruning techniques can prune out the layer weights; however, they cannot fully address the high computation complexity of GNN inference, caused by large graph size and complicated node connections.
In this paper, we propose an Iterative Gradient Rank Pruning (IGRP) algorithm to find graph lottery tickets (GLT) of GNNs where each GLT includes a pruned adjacency matrix and a sub-network. Our IGRP can avoid layer collapse and the winning ticket achieves Maximal critical compression. We evaluate the proposed method on small-scale (Cora and Citeseer), medium-scale (PubMed and Wiki-CS), and large-scale (Ogbn-ArXiv and Ogbn-Products) graph datasets. We demonstrate that both Single-shot and Multi-shot of IGRP outperform the state-of-the-art unified GNN sparsification (UGS) framework on node classification.

## Methodology
![Gradient Score Based Pruning](https://user-images.githubusercontent.com/42706378/204112992-a59a88b0-aa0e-4598-a853-97ceda145a65.png)
An illustration of the Multi-shot Iterative Gradient Rank Pruning (IGRP) Framework. Dash/solid lines denote the removed/remaining edges and weights in the
graph and GNNs, respectively.

## Datasets
CORA, CITESEER, PUBMED, WIKI-CS, OGBN-ARXIV, OGBN-PRODUCTS

## Experiments
Singleton Graph and Weight Lottery Ticket Identification
Adjacency Matrix and Weight Pruning
Unified Graph Lottery Ticket Identification
Multi-shot Iterations
Score Order Selection
Gradient Rank Selection
