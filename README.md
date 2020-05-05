# edhsmm
An **incomplete** implementation of Explicit Duration Hidden Semi-Markov Model (EDHSMM) in Python 3.

This exists because I attempted to improve a project, but because of the COVID-19 pandemic and time constraint, I decided to not apply this anymore. Anyways, I will still try to complete this in the future.

I have very little knowledge in software development, so this is where I need most help.

The EM algorithm is based on [Yu(2010)](https://www.sciencedirect.com/science/article/pii/S0004370209001416), while the Viterbi algorithm will be based on [Benouareth et al.(2008)](https://link.springer.com/article/10.1155/2008/247354).

The code style is based on [hmmlearn](https://github.com/hmmlearn/hmmlearn) and [jvkersch/hsmmlearn](https://github.com/jvkersch/hsmmlearn).

I implemented this on [Anaconda](https://www.anaconda.com/products/individual) (Python 3) on Windows (see the [notebooks](https://github.com/poypoyan/edhsmm/tree/master/notebooks)).

#### Implemented so far
- EM Algorithm (with & without right-censoring) 
- EDHSMM with Gaussian Emissions
- Scoring (log-likelihood of observation under the model)

#### To be Implemented
- Viterbi Algorithm
- EM Algorithm with multiple observation sequences
- Cythoning the core algorithms
- Generate samples
- *and a lot more* (to be posted in [issues](https://github.com/poypoyan/edhmm/issues))
