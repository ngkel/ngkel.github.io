---
layout: distill
title: Backpropagation
description: 
tags: optimization
giscus_comments: true
date: 2025-09-02
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
  - name: Introduction to ISTA
  - name: Convex Functions
  - name: Gradient Descent
  - name: Subgradient Method
  - name: Proximal Gradient Descent
  - name: How solution to LASSO inspire more efficient solution to basis pursuit
  - name: Connection between ISTA and deep network

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
pseudocode: true
---
# Motivations of backpropagation

## From ISTA to LISTA

ISTA is not a learning problem but a sparse coding problem. It is run on sample level data. The problem is also relatively simple and we can derive the update steps by hand and compute the gradient. 

LISTA, on the other hand, when compute the loss function, we said the computation of the loss function is completed only when we calculate the loss over the entire dataset. Thus we have discussion on batch size.



