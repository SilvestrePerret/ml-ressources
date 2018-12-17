
<h1><center>Readme : Machine Learning Cheatsheet</center></h1>

<h1>Table of Contents<span class="tocSkip"></span></h1>
<div class="toc"><ul class="toc-item"><li><span><a href="#Define-the-problem" data-toc-modified-id="Define-the-problem-1"><span class="toc-item-num">1&nbsp;&nbsp;</span>Define the problem</a></span><ul class="toc-item"><li><span><a href="#What-is-the-problem-?" data-toc-modified-id="What-is-the-problem-?-1.1"><span class="toc-item-num">1.1&nbsp;&nbsp;</span>What is the problem ?</a></span></li><li><span><a href="#Why-does-the-problem-need-to-be-solved-?" data-toc-modified-id="Why-does-the-problem-need-to-be-solved-?-1.2"><span class="toc-item-num">1.2&nbsp;&nbsp;</span>Why does the problem need to be solved ?</a></span></li><li><span><a href="#How-should-I-solve-the-problem-?" data-toc-modified-id="How-should-I-solve-the-problem-?-1.3"><span class="toc-item-num">1.3&nbsp;&nbsp;</span>How should I solve the problem ?</a></span></li></ul></li><li><span><a href="#Prepare-the-data" data-toc-modified-id="Prepare-the-data-2"><span class="toc-item-num">2&nbsp;&nbsp;</span>Prepare the data</a></span><ul class="toc-item"><li><span><a href="#Select-Data" data-toc-modified-id="Select-Data-2.1"><span class="toc-item-num">2.1&nbsp;&nbsp;</span>Select Data</a></span></li><li><span><a href="#Preprocess-Data" data-toc-modified-id="Preprocess-Data-2.2"><span class="toc-item-num">2.2&nbsp;&nbsp;</span>Preprocess Data</a></span></li><li><span><a href="#Transform-Data-(Features-Engineering)" data-toc-modified-id="Transform-Data-(Features-Engineering)-2.3"><span class="toc-item-num">2.3&nbsp;&nbsp;</span>Transform Data (Features Engineering)</a></span></li></ul></li><li><span><a href="#Spot-check-algorithms" data-toc-modified-id="Spot-check-algorithms-3"><span class="toc-item-num">3&nbsp;&nbsp;</span>Spot-check algorithms</a></span><ul class="toc-item"><li><span><a href="#DEFINE-A-TEST-HARNESS-FIRST" data-toc-modified-id="DEFINE-A-TEST-HARNESS-FIRST-3.1"><span class="toc-item-num">3.1&nbsp;&nbsp;</span>DEFINE A TEST HARNESS FIRST</a></span></li><li><span><a href="#THEN-SPOT-CHECK-MANY-ALGORIHTMS" data-toc-modified-id="THEN-SPOT-CHECK-MANY-ALGORIHTMS-3.2"><span class="toc-item-num">3.2&nbsp;&nbsp;</span>THEN SPOT CHECK MANY ALGORIHTMS</a></span></li></ul></li><li><span><a href="#Improve-results" data-toc-modified-id="Improve-results-4"><span class="toc-item-num">4&nbsp;&nbsp;</span>Improve results</a></span><ul class="toc-item"><li><span><a href="#Algorithms-tuning" data-toc-modified-id="Algorithms-tuning-4.1"><span class="toc-item-num">4.1&nbsp;&nbsp;</span>Algorithms tuning</a></span></li><li><span><a href="#Ensembles-method" data-toc-modified-id="Ensembles-method-4.2"><span class="toc-item-num">4.2&nbsp;&nbsp;</span>Ensembles method</a></span></li><li><span><a href="#Features-Engineering" data-toc-modified-id="Features-Engineering-4.3"><span class="toc-item-num">4.3&nbsp;&nbsp;</span>Features Engineering</a></span></li></ul></li><li><span><a href="#Present-results" data-toc-modified-id="Present-results-5"><span class="toc-item-num">5&nbsp;&nbsp;</span>Present results</a></span></li></ul></div>

## Define the problem

### What is the problem ?
- informal description : explain it to a friend


- formal description :
    <center style="padding: 10px 100px;">_A computer program is said to learn from experience E with respect to some class of tasks T and performance measure P, if its performance at tasks in T, as measured by P, improves with experience E._</center>
    For example :
    - Task T : Classify a tweet as "going to be retweeted" or not 
    - Experience E : A corpus of tweets for an account with both classes
    - Performance P : Classification accuracy, the percentage of well predicted tweets out of all tweets


- assumptions : make a list of assumptions about the problem 
    - Precise which one can be __tested__ against the data
    - Precise which one need to be __challenged__


- similar problems : Make a list of similar problems. This can guide you to new algo or concept (as model drift over time).

### Why does the problem need to be solved ?
- Motivation : Make it clear
    - Wanting to learn $\ne$ risking to be fired.


- Solution benefits : Make the benefits for you/your team clear


- Solution use : Use case ? Lifetime of the solution ?

### How should I solve the problem ?
List out step-by-step:
- which data you would collect and __how to prepare it__
- which prototypes and experiments you have to do to highlight questions (or assumptions)

## Prepare the data

See [scikit doc](https://scikit-learn.org/stable/modules/preprocessing.html#preprocessing).

### Select Data
__Know__ the scope of available data (__no code here__) :
- Time range ? Geographical ? Classes proportion
- What is __missing__ (and needed) ?
- What is __unnecessary__ ?

### Preprocess Data 
- __Formatting__ : relational database $\leftrightarrow$ text file $\leftrightarrow$ proprietary format


- __Cleaning__ : 
    - Deal with the __missing__ data
    - Remove the __unnecessary__ data
    - Anonymize/Pseudonymize


- __Detect and deal with outliers__ :
    - Detection method :
        - extreme value analysis (statistical tails, z-score)
        - probabilistic model (model the data with a gaussian mixture distribution and explore points with very little probability appearance)
        - linear model (PCA on the data, points with large residual error may be outliers)
        - proximity-based model (cluster the data, points which are isolated from the mass may be outliers)
        - information theoric model (outliers are points which increase the complexity (i.e. minimum code lenght) of the dataset.


- __Sampling__ :
    - Take a reasonable amount of data to work with. __Care about representativity__.

### Transform Data (Features Engineering)
You can spend a lot of time engineering features.__Start small and build on__

- __Scaling/reducing__ 


- __Decomposition__ : example: Datetime $\rightarrow$ Date & Time separated fields


- __Aggregation__ : example: One row per client $\rightarrow$ One row per day

## Spot-check algorithms
__Main objectif: Rapidly test algorithms__ to test whether or not there is a structure to learn in your problem __and__ which algorithms are effective.

### DEFINE A TEST HARNESS FIRST
- Performance measure: See [scikit doc](https://scikit-learn.org/stable/modules/model_evaluation.html)
    - classification, regression, clustering, ...
        - classification : [WIP]
        - regression :
            - $r^2$ score
            - Mean Absolute Error (MAE)
            - Mean Squared Error (MSE)
            - ...
        - clustering : [WIP]
    - problem specific :
        - false positive/negative
        - under/overestimation

- Split dataset into train and test (beginner):
    - 66% train, 34% test
    
- Cross validation (if more confident) :
    - 3, 5, 7, 10 folds $\rightarrow$ find a balance between size and representation
    
### THEN SPOT CHECK MANY ALGORIHTMS
- First algo to check is __random__ (also see `DummyClassifier` and `DummyRegressor` from `scikit-learn`)


- Then choose 5-10 __standard__ and __appropriate__ algorithms to spot check (from [here](http://machinelearningmastery.com/a-tour-of-machine-learning-algorithms/)).

## Improve results
Also see this [cheatsheet](https://machinelearningmastery.com/machine-learning-performance-improvement-cheat-sheet/).

### Algorithms tuning

__The mode tuned your algorithm is, the more biased the result will be $\rightarrow$ Risk of over-fitting__

See the [scikit doc](https://scikit-learn.org/stable/modules/grid_search.html).
- Grid search
- Random search 

### Ensembles method

See the [scikit doc](https://scikit-learn.org/stable/modules/grid_search.html#out-of-bag-estimates).

- __Bagging__ (same algo trained on different subsets, example : )
- __Boosting__ (different algo trained on same dataset, example : )
- __Blending__ (models ouputs are the input of new model to make the prediction, example : )

### Features Engineering

See scikit doc on [features extraction](https://scikit-learn.org/stable/modules/feature_extraction.html) and [features selection](https://scikit-learn.org/stable/modules/feature_selection.html).

It's recommended to perform this process __one step at a time__ and to __create a new test/train dataset for each modification__ you make and then test algorithms on the dataset.

## Present results

[WIP]
