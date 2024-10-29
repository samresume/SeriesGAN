
# SeriesGAN: Time Series Generation via Adversarial and Autoregressive Learning

## Abstract
Current Generative Adversarial Network (GAN)-based approaches for time series generation face challenges such as suboptimal convergence, information loss in embedding spaces, and instability. To overcome these challenges, we introduce an advanced framework that integrates the advantages of an autoencoder-generated embedding space with the adversarial training dynamics of GANs. This method employs two discriminators: one to specifically guide the generator and another to refine both the autoencoder's and generator's output. Additionally, our framework incorporates a novel autoencoder-based loss function and supervision from a teacher-forcing supervisor network, which captures the stepwise conditional distributions of the data. The generator operates within the latent space, while the two discriminators work on latent and feature spaces separately, providing crucial feedback to both the generator and the autoencoder. By leveraging this dual-discriminator approach, we minimize information loss in the embedding space. Through joint training, our framework excels at generating high-fidelity time series data, consistently outperforming existing state-of-the-art benchmarks both qualitatively and quantitatively across a range of real and synthetic multivariate time series datasets.

<img src="seriesgan.svg" width="600" alt="SeriesGAN Architecture" title="SeriesGAN Architecture">


## How to Cite
The paper associated with this repository has been accepted at BigData 2024 as a regular paper for oral presentation. We kindly ask you to provide a citation to acknowledge our work.

Available on arXiv:
[https://arxiv.org/abs/2410.21203](https://arxiv.org/abs/2410.21203)

Here is the BibTeX citation for your reference:

 ```
@misc{eskandarinasab2024seriesgan,
      title={SeriesGAN: Time Series Generation via Adversarial and Autoregressive Learning}, 
      author={MohammadReza EskandariNasab and Shah Muhammad Hamdi and Soukaina Filali Boubrahimi},
      year={2024},
      eprint={2410.21203},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2410.21203}, 
}
```
We appreciate your citation!

## Installation
Clone the repository and install dependencies:
```bash
git clone https://github.com/samresume/SeriesGAN.git
cd SeriesGAN
pip install -r requirements.txt
```

## Usage
To get started, run the tutorial notebook:
```bash
jupyter notebook tutorial.ipynb
```


## Files
- `seriesgan.py`: Main implementation of the SeriesGAN model.
- `data_loading.py`: Functions for loading and preprocessing data.
- `utils.py`: Helper utilities for the model.


## License

This project is licensed under the MIT License.
