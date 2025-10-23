---
layout: distill
title: What to learn and introduction to problems solvable by analytical approach
description: Idealistic models that inspire deep network structures
tags: representation_learning
giscus_comments: true
date: 2025-08-18
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
  - name: Learned ISTA

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
