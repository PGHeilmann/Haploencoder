## Data preparation

- preprocessing.R is used to download the marker and yield data from Technow et al. (2014). This is followed by some light filtering (data is already filterd) and the formation of haplotype blocks.
- hybridgen.R creates the hybrid marker data that is required for the semi-supervised autoencoder.

The genetic map is included in the folder *data* or can be downloaded via the original paper from Technow et al. (2014). The folder *data* also contains a filtered version of the markers as well as the assignment of markers to haplotype blocks ("LD07FL" => Linkage-based blocks, threshold = 0.7, flanking method). This allows users to skip using the R package SelectionTools as it is temporarily not accessible. This folder also contains the original cross validation splits in the file *TrainTestIndex.csv*. Y_data.csv contains target trait data as provided with the original dataset.

If the goal is to create and extract latent block variables to use them as inputs in Machine Learning, it is probably better to use the hybrid marker data as the original inputs. If the goal is to fit a GBLUP, it might be a better idea to compress the marker data of the parents as it is usually smaller in size. The compressed marker data for each parent can then be used to calculate genomic relationship matrices.

## Autoencoder

- classes.py contains all the custom classes required to run the autoencoder. This includes the neural network architecture, a custom dataset for easier dataloading and a custom correlation loss function.
- standalone_autoencoder_example.py contains a fully functional script to run the semisupervised autoencoder with the given public dataset of Technow et al. (2014).
