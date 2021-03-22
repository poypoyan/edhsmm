# edhsmm
An(other) implementation of Explicit Duration Hidden Semi-Markov Model (EDHSMM) in Python 3.

This exists because I attempted to improve a project, but because of the COVID-19 pandemic, I decided to not apply this anymore. Anyways, I will still try to work on this in the future.

I have very little knowledge in software development, so this is where I need most help.

The EM algorithm is based on [Yu (2010)](https://www.sciencedirect.com/science/article/pii/S0004370209001416), while the Viterbi algorithm is based on [Benouareth et al. (2008)](https://link.springer.com/article/10.1155/2008/247354).

The code style is inspired from [hmmlearn](https://github.com/hmmlearn/hmmlearn) and [jvkersch/hsmmlearn](https://github.com/jvkersch/hsmmlearn).

I implemented this on [Anaconda](https://www.anaconda.com/products/individual) (Python 3) on Windows. I need testing for other platforms.

#### Implemented so far
- EM algorithm (with & without right-censoring)
- Scoring (log-likelihood of observation under the model)
- Viterbi algorithm
- Generate samples
- Support for multivariate Gaussian emissions
- Support for multiple observation sequences
- Support for multinomial (discrete) emissions

#### Sure to be implemented
- ~~Viterbi Algorithm~~ (06-May-2020)
- ~~EM Algorithm with multiple observation sequences~~ (31-Dec-2020)
- ~~Cythoning the core algorithms~~ (25-Jun-2020) (see the [hsmm_core_x.pyx](edhsmm/hsmm_core_x.pyx) and [setup.py](edhsmm/setup.py))
- ~~Generate samples~~ (06-Dec-2020)
- ~~Customizable duration distribution~~ (12-Jun-2020)
- ~~MultinomialHSMM (discrete emissions)~~ (23-Mar-2021)

#### Unsure to be implemented
- Left-censoring (I have difficulty understanding and implementing it)

#### Installation & Tutorial
For now, there is no installation; you need to download the ZIP for this repository. For tutorial, see the [notebooks](notebooks).

Found a bug? Suggest a feature? Please post on [issues](https://github.com/poypoyan/edhmm/issues). 😊
