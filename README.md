<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="assets/logo_dark.svg">
    <source media="(prefers-color-scheme: light)" srcset="assets/logo_light.svg">
    <img alt="BOOMER - Gradient Boosted Multi-Label Classification Rules" src="assets/logo_light.svg">
  </picture>
</p>

> [!NOTE]
> This repository is now archived and has been replaced with [https://github.com/mrapp-ke/MLRL-Boomer](https://github.com/mrapp-ke/MLRL-Boomer). 

This software package provides the official implementation of **BOOMER - an algorithm for learning gradient boosted multi-label classification rules** that integrates with the popular [scikit-learn](https://scikit-learn.org) machine learning framework.

The goal of [multi-label classification](https://en.wikipedia.org/wiki/Multi-label_classification) is the automatic assignment of sets of labels to individual data points, for example, the annotation of text documents with topics. The BOOMER algorithm uses [gradient boosting](https://en.wikipedia.org/wiki/Gradient_boosting) to learn an ensemble of rules that is built with respect to a given multivariate loss function. To provide a versatile tool for different use cases, great emphasis is put on the *efficiency* of the implementation. To ensure its *flexibility*, it is designed in a modular fashion and can therefore easily be adjusted to different requirements.

