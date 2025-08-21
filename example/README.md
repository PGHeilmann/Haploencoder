- Preprocessing.R is used to download the marker and yield data from Technow et al. (2014). This is followed by some light filtering (data is already filterd) and the formation of haplotype blocks.
- Hybridgen.R creates the hybrid marker data that is required for the semi-supervised autoencoder.

The genetic map is included in the folder *data* or can be downloaded via the original paper from Technow et al. (2014). The folder *data* also contains a filtered version of the markers as well as the assignment of markers to haplotype blocks ("LD07FL" => Linkage-based blocks, threshold = 0.7, flanking method). This allows users to skip using the R package SelectionTools as it is temporarily not accessible.

