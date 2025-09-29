## Data Preparation

- preprocessing.R downloads the marker and yield data from Technow et al. (2014). This is followed by light filtering (the data is already filtered) and the formation of haplotype blocks.
- hybridgen.R creates the hybrid marker data required for the semi-supervised autoencoder.

The genetic map is included in the data folder or can be downloaded via the original paper from Technow et al. (2014). The data folder also contains a filtered version of the markers as well as the assignment of markers to haplotype blocks ("LD07FL" = linkage-based blocks, threshold = 0.7, flanking method). This allows users to skip using the R package SelectionTools, which is temporarily unavailable. The folder also includes the original cross-validation splits in TrainTestIndex.csv. Y_data.csv contains target trait data as provided with the original dataset.

If the goal is to create and extract latent block variables for use as inputs in machine learning, it is generally better to use the hybrid marker data as the original inputs. If the goal is to fit a GBLUP, it may be preferable to compress the parental marker data, as it is usually smaller in size. The compressed marker data for each parent can then be used to calculate genomic relationship matrices.

## Autoencoder

- classes.py contains all the custom classes required to run the autoencoder. This includes the neural network architecture, a custom dataset class for easier data loading, and a custom correlation loss function.
- standalone_autoencoder_example.py contains a fully functional script to run the semi-supervised autoencoder with the public dataset from Technow et al. (2014).
