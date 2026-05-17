---
layout: distill
title: Linear probing and subspace classification
description: 
tags: representation_learning
giscus_comments: true
date: 2026-03-08
featured: true
mermaid:
  enabled: true
  zoomable: true
code_diff: true
map: true
chart:
  chartjs: true
  echarts: true
  vega_lite: true
tikzjax: true
typograms: true

authors:
  - name: Ng Ka Lok
    url: "ngkel.github.io"
    affiliations:
      name: Ex-HKU, Foodpanda

bibliography: 2018-12-22-distill.bib

# Optionally, you can add a table of contents to your post.
# NOTES:
#   - make sure that TOC names match the actual section names
#     for hyperlinks within the post to work correctly.
#   - we may want to automate TOC generation in the future using
#     jekyll-toc plugin (https://github.com/toshimaru/jekyll-toc).
toc:
  - name: What learning is all about? Or at least, what are modern methods actually doing?
  - name: Learning problems that are solvable by analytical approach
  # if a section has subsections, you can add them as follows:
    subsections:
      - name: PCA - finding the single subspace that best fits the data
      - name: Power Iteration - Workhorse of PCA
      - name: Limitation of PCA
      - name: Mixtures of Subspaces and Sparsely-Used Dictionaries
      - name: Overcomplete dictionary learning

# Below is an example of injecting additional post-specific styles.
# If you use this post as a template, delete this _styles block.
_styles: >
  .fake-img {
    background: #bbb;
    border: 1px solid rgba(0, 0, 0, 0.1);
    box-shadow: 0 0px 4px rgba(0, 0, 0, 0.1);
    margin-bottom: 12px;
  }
  .fake-img p {
    font-family: monospace;
    color: white;
    text-align: left;
    margin: 12px 0;
    text-align: center;
    font-size: 16px;
  }
---

# Introduction

There are 2 key inspirational ideas from LeJEPA paper. 

1. The claim that isotropic Gaussian is the property of optimal embedding to minimize downstream prediction risk across broad task families, both linear and non-linear. Thus, transforming data to features distributed as isotropic Gaussian is suggested as the objective of foundation model.
2. SIGReg, an objective function that promote isotropic Gaussian features.

This blog is more on discussion for the first idea. I want to discuss the choice of evaluation of the encoder. The reason is that I have read about 2 objectives in representation learning that are claimed "general" for self-supervised learning. Isotropic Gaussian vs linear discriminative representation. They claim their victory by showcasing the learnt features being able to solve some downstream tasks. If our aim is to build a self-supervised learning system that helps solving unknown downstream task, which property should be optimal? 

Another note-worthy thing is that, both 

2 questions:

1. What makes a good evaluation method
2. Is there really only one "optimal" evaluation method?

# Linear probing 

# Subspace classification

# Non-linear probing

# LeJEPA shows interesting temporal straightening effect

# Can LDR show similar temporal straightening effect?

# Can LeJEPA explain the variety of data like LDR?

# Quick thought
LDR form a good coordinate system where the top/global level features can live in and we demand the global features for decision making to be isotropic.

In image classification task, instead of saying that CLS embedding containing "global" information, we may say that it contains task relevant information.
