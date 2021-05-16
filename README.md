# edhsmm
An(other) implementation of Explicit Duration Hidden Semi-Markov Model (EDHSMM) in Python 3.

The EM algorithm is based on [Yu (2010)](https://www.sciencedirect.com/science/article/pii/S0004370209001416), while the Viterbi algorithm is based on [Benouareth et al. (2008)](https://link.springer.com/article/10.1155/2008/247354).

The code style is inspired from [hmmlearn](https://github.com/hmmlearn/hmmlearn) and [jvkersch/hsmmlearn](https://github.com/jvkersch/hsmmlearn).

**Note:** This is the "main" branch, which is for release. Use the "test" branch for prototyping.

#### Implemented so far
- EM algorithm (with & without right-censoring)
- Scoring (log-likelihood of observation under the model)
- Viterbi algorithm
- Generate samples
- Support for multivariate Gaussian emissions
- Support for multiple observation sequences
- Support for multinomial (discrete) emissions

See the README in the [test](https://github.com/poypoyan/edhsmm/tree/test) branch for the full timeline.

#### Dependencies
- python >= 3.5
- numpy >= 1.10
- scikit-learn >= 0.16
- scipy >= 0.19

#### Installation & Tutorial
```console
pip install edhsmm
```
For tutorial, see the [notebooks](notebooks).

Found a bug? Suggest a feature? Please post on [issues](https://github.com/poypoyan/edhmm/issues).