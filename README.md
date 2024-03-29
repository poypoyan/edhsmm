**Warning:** I made this repo when I was an undergrad, but was not even part of my undergrad project. Correctness of implementation not guaranteed, so use at your own risk.

# edhsmm
An(other) implementation of Explicit Duration Hidden Semi-Markov Models in Python 3

The EM algorithm is based on [Yu (2010)](https://www.sciencedirect.com/science/article/pii/S0004370209001416) (Section 3.1, 2.2.1 & 2.2.2), while the Viterbi algorithm is based on [Benouareth et al. (2008)](https://link.springer.com/article/10.1155/2008/247354).

The code style is inspired from [hmmlearn](https://github.com/hmmlearn/hmmlearn) and [jvkersch/hsmmlearn](https://github.com/jvkersch/hsmmlearn).

#### Implemented so far
- EM algorithm
- Scoring (log-likelihood of observation under the model)
- Viterbi algorithm
- Generate samples
- Support for multivariate Gaussian emissions
- Support for multiple observation sequences
- Support for multinomial (discrete) emissions

#### Dependencies
- python >= 3.6
- numpy >= 1.17
- scikit-learn >= 0.16
- scipy >= 0.19

#### Installation & Tutorial
Via *PyPI*:
```console
pip install edhsmm
```

Via *source*:
```console
pip install .
```

Test in *venv*:

Windows
```console
git clone https://github.com/poypoyan/edhsmm.git
cd edhsmm
python -m venv edhsmm-venv
edhsmm-venv\Scripts\activate
pip install --upgrade -r requirements.txt
pip install .
```
Type `python` to run Python CLI, and type `deactivate` to exit the environment.

Linux
```console
git clone https://github.com/poypoyan/edhsmm.git
cd edhsmm
python3 -m venv edhsmm-venv
source edhsmm-venv/bin/activate
pip install --upgrade -r requirements.txt
pip install .
```
Type `python3` to run Python CLI, and type `deactivate` to exit the environment.

**Note**: Also run `pip install notebook matplotlib` to run the notebooks.

For tutorial, see the [notebooks](notebooks). This also serves as some sort of "documentation".

Found a bug? Suggest a feature? Please post on [issues](https://github.com/poypoyan/edhmm/issues).
