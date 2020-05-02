# edhmm
An **incomplete** implementation of Explicit Duration Hidden Markov Model (EDHMM) in Python 3. Supports right-censoring, but no left-censoring.

This exists because I attempted to "improve" a project, but because of the COVID-19 pandemic and time constraint, I decided to not apply this anymore. Anyways, I will still try to complete this in the future.

The algorithms are based on [Yu(2010)](https://www.sciencedirect.com/science/article/pii/S0004370209001416).

The code style is based on [hmmlearn](https://github.com/hmmlearn/hmmlearn) and [jvkersch/hsmmlearn](https://github.com/jvkersch/hsmmlearn).

#### Implemented so far
- EM Algorithm (with & without right-censoring) 
- EDHMM with Gaussian Emissions
- Scoring (log-likelihood of observation under the model)

#### To be Implemented
- Viterbi Algorithm
- EM Algorithm with Multiple Observations
- Cythoning the core algorithms
- Generate samples
- *and* a lot more (to be posted in *issues*)
