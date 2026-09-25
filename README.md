# AALpy-extra

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)

**AALpy-extra** is an extension library for [AALpy](https://github.com/DES-Lab/AALpy) —
an active automata learning library.  It provides a community space for algorithms, tools, and integrations that extend 
AALpy, while **AALpy** itself provides a space for stable and lightweight learning algorithms and basic auxiliary functions
 At the same time, we follow AALpy's style of implementation and documentation. 

## Why a Separate Repository?

AALpy strives to be lightweight and dependency-minimal (its only runtime
dependency is `pydot`).  Some useful extensions, however, bring heavy or
platform-specific dependencies. **AALpy-extra** provides a space for extensions that are useful for the community but 
may be dependency-heavy or more experimental.
The idea is similar to the relationship between [Stable-Baselines3](https://stable-baselines3.readthedocs.io/) 
and [SB3-Contrib](https://stable-baselines3.readthedocs.io/en/master/guide/sb3_contrib.html): AALpy contains the lightweight core, 
while AALpy-extra provides additional functionality that does not need to be part of the core distribution.

Examples include:
- **stormpy integration** ([Storm](https://www.stormchecker.org/) model checker bindings), which 
  requires native C++ libraries (Storm) and is only available as a binary wheel on select platforms.
- Future extensions such as black-box checking would add
  similarly substantial native dependencies.

Forcing these on every AALpy user would defeat the library's design goals.
AALpy-extra solves this by making such integrations available as **optional
extensions**. You install only what you need.

## Installation

AALpy-extra is not yet published on PyPI — installation from the Python
Package Index will follow soon.  Until then, install it from the source
repository:

```bash
git clone https://github.com/zwergziege/AALpy-extra.git
cd AALpy-extra
pip install .
```

This installs AALpy-extra and its sole hard dependency, AALpy.

### Editable / Development Install

For development, install in editable mode so changes to the source are
picked up without reinstalling:

```bash
pip install -e .
```

### With Storm Model Checking

```bash
pip install ".[stormpy]"
```

Installs AALpy-extra together with `stormpy`, enabling the Storm model
checking interface.

### With SAT-Based Learning

```bash
pip install ".[python-sat]"
```

Installs AALpy-extra together with `python-sat`, enabling the SAT-based
incomplete-teacher learner.

### Everything

```bash
pip install ".[all]"
```

Installs the base library plus all optional dependencies.

## Features

### Storm Model Checking (`aalpy_extra.utils.storm_interface`)

Convert AALpy automata models to **stormpy** sparse models and perform
probabilistic model checking on AALpy MDPs and SMMs:

- `mdp_to_stormpy(mdp)` — Convert an AALpy MDP to a stormpy `SparseMdp`
- `smm_to_stormpy_mdp(smm)` — Convert a Stochastic Mealy Machine to a stormpy MDP
- `stormpy_model_check(model, property_string)` — Check a single property
- `stormpy_model_check_properties(model, properties)` — Batch model checking

#### Quick Start 

```python
from aalpy.automata import Mdp, MdpState
from aalpy_extra.utils.storm_interface import mdp_to_stormpy, stormpy_model_check

# Create an AALpy MDP
s0 = MdpState("s0", "start")
s1 = MdpState("s1", "goal")
s0.transitions["a"].append((s1, 1.0))
mdp = Mdp(s0, [s0, s1])

# Convert to a stormpy sparse model
stormpy_model = mdp_to_stormpy(mdp)

# Model check: probability of reaching the "goal" state
result = stormpy_model_check(stormpy_model, 'Pmax=? [F "goal"]')
print(f"Probability of reaching goal: {result:.4f}")
# Probability of reaching goal: 1.0000
```

You can also pass AALpy models directly to the model checking functions —
conversion happens transparently:

```python
result = stormpy_model_check(mdp, 'Pmax=? [F "goal"]')
```
For syntax of the PRISM language and properties, 
please refer to the documentation of [Storm](https://www.stormchecker.org/)
and [PRISM](https://www.prismmodelchecker.org/).

## Documentation

- [AALpy documentation](https://github.com/DES-Lab/AALpy) — the base library
- [stormpy documentation](https://moves-rwth.github.io/stormpy/) — the Storm model checker Python bindings

## Contributing

Contributions are welcome!  If you have an algorithm, tool integration, or
utility that builds on AALpy but doesn't quite fit the core library, this
is the place for it.

Please follow AALpy's coding conventions:
- Google-style docstrings with `:param:` and `:return:` sections
- Type hints for all public function signatures
- Lazy imports for optional dependencies (so users without the dependency
  get a clear error message only when they try to use the feature)

## Citing

If you use AALpy-extra in your research, please cite the underlying AALpy
library:

- [Extended version (preferred)](https://www.researchgate.net/publication/359517046_AALpy_an_active_automata_learning_library/citation/download)
- [Tool paper](https://dblp.org/rec/conf/atva/MuskardinAPPT21.html?view=bibtex)

Additionally, in case you use one of the available extensions, please check the documentation 
for reference to a potentially published paper.


## License

MIT License — see [LICENSE](LICENSE).
