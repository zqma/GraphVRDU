# LayoutGNN
## Content
This repo is the implementation of Layout GNN for document immge classification
The dataset used is [RVL-CDIP](https://www.cs.cmu.edu/~aharley/rvl-cdip/)
The model infomation is available on [spektral](https://graphneural.network/layers/convolution/)


## Highlight
a novel graph readout function by combining edge and node feature.

## Usage
1. generate_data.py will save the images as jpg
2. parse.py will run ocr on the images and save the outputs
3. generate_embedding will apply language model/ visual model to embed features
4. graph_dataset.py will create a dataset of graph objects
5. Finally, gcn.py will run gcns
