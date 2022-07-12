# Getting Start with Recommender System


The Variational Autoencoder (VAE) shown here is an optimized implementation of the architecture first described in Variational Autoencoders for Collaborative Filtering and can be used for recommendation tasks. The main differences between this model and the original one are the performance optimizations, such as using sparse matrices, mixed precision, larger mini-batches and multiple GPUs. These changes enabled us to achieve a significantly higher speed while maintaining the same accuracy. Because of our fast implementation, we've also been able to carry out an extensive hyperparameter search to slightly improve the accuracy metrics.

By using Variational Autoencoder for Collaborative Filtering (VAE-CF), you can quickly train a recommendation model for the collaborative filtering task. The required input data consists of pairs of user-item IDs for each interaction between a user and an item. With a trained model, you can run inference to predict what items a new user most likely to interact with.

This model is trained with mixed precision using Tensor Cores on NVIDIA Volta, Turing and Ampere GPUs. Therefore, researchers can get results 1.9x faster than training without Tensor Cores, while experiencing the benefits of mixed precision training. This model is tested against each NGC monthly container release to ensure consistent accuracy and performance over time.

## Model


The model consists of two parts: the encoder and the decoder. The encoder transforms the vector, which contains the interactions for a specific user, into a n-dimensional variational distribution. We can use this variational distribution to obtain the latent representation of a user. This latent representation is then fed into the decoder. The result is a vector of item interaction probabilities for a particular user.

The dataset that we use for training this model is a movie rating dataset. The trained model predicts the rate of a movie for a user based on previous rates of the user for other movies. It predicts what is the probability that the user choose a movie.

## Default Configuration
The following features were implemented in this model:

- > Sparse matrix support
- > Data-parallel multi-GPU training
- > Dynamic loss scaling with backoff for tensor cores (mixed precision) training
- > Feature support matrix

### The following features are supported by this model:

Feature	VAE-CF
Horovod Multi-GPU (NCCL)	Yes
Automatic mixed precision (AMP)	Yes

## Requirements
This repository contains Dockerfile which extends the Tensorflow NGC container and encapsulates some dependencies. Aside from these dependencies, ensure you have the following components:

- > NVIDIA Docker
- >TensorFlow NGC container
- > Supported GPUs:
> - > NVIDIA Volta architecture
> - > NVIDIA Turing architecture
> - > NVIDIA Ampere architecture

Follow the rest of the process from building the container to train and test the model in quick start guide section.
