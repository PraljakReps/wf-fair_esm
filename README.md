Evolutionary Scale Modeling (ESM) embeds biologically meaningful representations for protein sequences using pretrained Transform protein language models from Facebook AI research.
   
# Evolutionary Scaling Modeling

This package provides an implementation of the inference pipeline of Evolutionary Scale Modeling from Facebook AI research. This is a pretrained Transformer protein language model on millions of homologous protein sequences. Each pretrained Transformer model (ESM) allows for inferring meaningful protein sequence representations that are biologically meaningful. For more details, please check out the paper by Facebook AI research: [Biological structure and function emerge from scaling unsupervised learning to 250 million protein sequences" (Rives et al., 2019)](https://doi.org/10.1101/622803). These pretrained models are excellent starting points for downstream tasks, ranging from predicting mutant variant effects to contact prediction.


## Model parameters on Latch Bio

Here, I am will walk through an example of choosing the input parameters for inferring representations of a protein library.
Protein data csv Dir
```
# file that corresponds to the the protein library in a .csv file
# make sure the first and second column corresponds to header names (e.g. protein_1, ..., protein_XX) and primary sequence (e.g. WTAS..SQ)
/esm/lactamase/lactamase_protein_dataset.csv
```

Output directory containing the protein representations in a .csv format
```
# output directory containing the output files
temp/esm_output/
```
   
What is the name of the protein of interest?
```
# name the output folder accordining the protein library (e.g. lactamase family)
lactamase
```
     
Pick pretrained model
```
# choose which pretrained Transformer you would like to download and implement.
esm1_t6_43M_UR50S
```
     
Pick hyperparameter value (only positive integer values).
```
# tsne hyperparameter for mapping higher dimensional representations to lower dimensional representations for visualization
30 
```
    
## Model output files
The outputs will be saved in a directory provided by the user (default: latch:///). The outputs include the computed mean protein represetnations, tsne lower dimensional representations in a .csv file, and the corresponding tsne plot of the representations as .png file. 
	* `{protein_name}_mean_seq_reprs.csv` - .csv file that return the mean representation vector for each input protein sequence and phenotype score.
	* `tsne_{protein_name}_mean_seq_reprs.csv` - .csv file that returns the tsne embeddings of the above representations.
	* `plot_{protein_name}_mean_seq_reprs.png` - .png file containg the tsne plot of the mean embeddings. 

## Available Pretrained Models on Latch SDK

| Shorthand | `esm.pretrained.` | #layers | #params | Dataset | Embedding Dim |  Model URL (automatically downloaded to `~/.cache/torch/hub/checkpoints`) |
|-----------|---------------------|---------|---------|---------|---------------|-----------------------------------------------------------------------|
| ESM-1b    | `esm1b_t33_650M_UR50S` | 33     | 650M    | UR50/S 2018_03 | 1280          | https://dl.fbaipublicfiles.com/fair-esm/models/esm1b_t33_650M_UR50S.pt   |
| ESM-1     | `esm1_t34_670M_UR50S` | 34      | 670M    | UR50/S 2018_03 | 1280          |  https://dl.fbaipublicfiles.com/fair-esm/models/esm1_t34_670M_UR50S.pt |
|           | `esm1_t34_670M_UR50D` | 34      | 670M    | UR50/D 2018_03 | 1280          |  https://dl.fbaipublicfiles.com/fair-esm/models/esm1_t34_670M_UR50D.pt |
|           | `esm1_t34_670M_UR100` | 34      | 670M    | UR100 2018_03  | 1280          |  https://dl.fbaipublicfiles.com/fair-esm/models/esm1_t34_670M_UR100.pt |
|           | `esm1_t12_85M_UR50S`  | 12      | 85M     | UR50/S 2018_03 | 768           |  https://dl.fbaipublicfiles.com/fair-esm/models/esm1_t12_85M_UR50S.pt  |
|           | `esm1_t6_43M_UR50S`   | 6       | 43M     | UR50/S 2018_03 | 768           |  https://dl.fbaipublicfiles.com/fair-esm/models/esm1_t6_43M_UR50S.pt   |
    

## Citations

If you use the code or data in this package, please cite:
    

```

@article{{rives2019biological,
        author={Rives, Alexander and Meier, Joshua and Sercu, Tom and Goyal, Siddharth and Lin, Zeming and Liu, Jason and Guo, Demi and Ott, Myle and Zitnick, C. Lawrence and Ma, Jerry and Fergus, Rob},
        title={Biological Structure and Function Emerge from Scaling Unsupervised Learning to 250 Million Protein Sequences},
        year={2019},
        doi={10.1101/622803},
        url={https://www.biorxiv.org/content/10.1101/622803v4},
        journal={bioRxiv}
            
    }

```



## License

The original ESM source code is liscense under the MIT license found in the `LISCENSE` file in the root directory of the following github repository: [facebookresearch/esm](https://github.com/facebookresearch/esm).

