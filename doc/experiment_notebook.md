# To keep track of observations about model training

## Initial experiments not yielding hypothesized results...

### Maybe L2 sentences are overweighted or too distracting

- Remedy options:

1. Weight dysarthric speech more than L2 speech at training time
2. Make sure that loss values are normalized by length, since losses for L2 might be higher

> To do: 
> 1. check gradients of loss function by tasks
> 2. check average loss by task

## Experiment to test different loss reduction methods

- Compared the losses between sum and mean reduction methods:
    - Mean divides the losses by sequence lengths
- Made sure that the model doesn't update (by matching batch size really high)
- The results are the losses *after* division but before the mean is taken

### Results:

#### Sum reduction:

Train (412 samples):

mean l2: 44.98539935541517 | mean dys: 28.582269360224405
pearson: PearsonRResult(statistic=0.4604387841523311, pvalue=5.1965427351233185e-23)
t-test: Ttest_indResult(statistic=-30.347448347919354, pvalue=2.6596968262464187e-136)

Train + Dev (480 samples):

mean l2: 97.85998662110347 | mean dys: 29.6677570658994
pearson: PearsonRResult(statistic=0.35107560349359557, pvalue=2.2762038725091704e-15)
t-test: Ttest_indResult(statistic=-13.926559234093528, pvalue=2.7755681088193772e-40)

#### Mean reduction:

Train (412 samples):

mean l2: 2.1760008982900114 | mean dys: 5.523369124418573
pearson: PearsonRResult(statistic=-0.3384758352082159, pvalue=2.0414437188500632e-13)
t-test: Ttest_indResult(statistic=-13.571706820336784, pvalue=2.850679708954572e-38)

Train + Dev (480 samples):

mean l2: 3.141343692495565 | mean dys: 5.740210406033389
pearson: PearsonRResult(statistic=-0.20973114235887813, pvalue=3.5755299246240973e-06)
t-test: Ttest_indResult(statistic=-15.512792605539468, pvalue=1.3772342842362518e-48)

## Comparing freezing feature extractor to not freezing

Keeping the learning rate constant, freezing the extractor seems to worsen performance:

### Frozen extractor (independent)

|Dataset   |Control|Dysarthric|
|----------|-------|----------|
|TORGO     |40.377 |71.924    |
|UA-Speech |26.231 |68.904    |

### No freezing (independent)

- 3% higher dysarthic scores (lower scores in most intelligibility levels)

|Dataset   |Control|Dysarthric|
|----------|-------|----------|
|TORGO     |32.749 |68.847    |
|UA-Speech |15.778 |60.043    |

### Weight decay

- Testing whether weight decay with a low learning over 25 epochs will help