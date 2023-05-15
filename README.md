# Deep TalRasha
*Unboxing basic deep learning models with minimal TF dependencies and off-the-shelf examples*

---

Feel fed-up with those heavy, nested and black-boxed deep learning projects? 
Every day sees people borrowing model codes and using them without knowing what's going on within the model. 
We reckon this is not very much a good signal to both the research community and industry.

A lot of people may already realize the following pain points in the realm of deep learning and AI research experiments:
* **Tons of dependencies:** So many recent research projects inherit from many open-sourced repos. Meeting these dependencies is usually time-consuming (and can easily have conflicts if some new add-ons are needed). This is not very friendly for one to play with the models or even make some modifications.
* **Black-boxed ops/models:** Nowadays, everything can be done with a single `import`. Who cares about what's going on inside the imported models/layers?
* **Lack of a hub of training examples:** We see a lot of checkpoint hubs, but it takes a while to source back to the implementation of training. 

All the problems above hinder us from knowing the theories behind the models, let alone customizing them. Is it all we can do to be a python package alchemist?

**No, we are going to master it.**

Here, we introduce this project `TalRasha` (initiated by myself though...It does not count so many people to make a 'we', sorry). What we are looking for is a repo that
* can run without installing a lot of packages;
* implements those complex models and operators as a white box;
* provides runnable examples of various models with suitable metrics and ready-to-use toy data;
* integrates these useful models and ops as a library for easy use.

We hope this project can help both junior AI researchers and engineers for learning, as well as the experienced ones for quick and convenient prototyping.


## 1 Requirements 
Currently, `TalRasha` only requires the following tensorflow packages and can be run on Windows, Linux and Mac without manual configuration.
```angular2html
tensorflow (v2)
tensorflow_hub
tensorflow_datasets
```
(but you'd better have at least one GPU)

At the moment, we are not considering packages such as `tensorflow_probability`, as employing ops from them contradicts our motivation of white-box models. We will implement those ops that are essential to the theoretical basis of the corresponding models, *e.g.*, [talrasha.layer.reparameterization](talrasha/layer/reparameterization.py) for ELBO-based models.

In the future, we may use [HuggingFace](https://huggingface.co/) packages to ease our work. 
This may happen when we need an arbitrary transformer-based backbone that is not very much related to the core of the model, 
or we need helpers and hubs of datasets/tokenizers. However, for easy learning and customization, we insist self-implementation of essential layers even if they are available in our dependencies.


## 2 Run Examples
See [examples](./examples). By default, an example comes with a Tensorboard event. Please check if you like during training.

## ... TBD

## -1 Contribute

Till now, this project is yet from my rough idea. I will allow PRs when CI/CD is ready.

In the case of a PR, let me leave some kind reminders.

### -1.1 Project Structure
* [examples](./examples): All examples of model training and experiments
* [talrasha](./talrasha): The op lib. In the future, I will make it installable.
  * [talrasha.functional](./talrasha/functional): Functional operations such as computing distance, distributional discrepancy or creating positional embeddings.
  * [talrasha.layer](./talrasha/layer): Literally the layers (tensor operators that sometimes include trainable/configurable members).
  * [talrasha.model](./talrasha/model): All the 'models' such as VAE, GAN, *etc*. It is suggested to use `keras.Model.add_loss()` to define a loss in the implementation.
  * [talrasha.util](./talrasha/util): Some helpers for data, training and evaluation.

### -1.2 Coding Style
See [here](https://github.com/google/styleguide/blob/gh-pages/pyguide.md), but I am not mastering it now.

Please leave docstrings (of the `reStructuredText` style) if necessary.




