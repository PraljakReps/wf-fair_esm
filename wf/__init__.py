"""
Evolutionary scaling model
"""

import subprocess
from pathlib import Path
import sys
from typing import Optional

from latch import small_task, large_task, workflow
from latch.types import LatchFile, LatchDir

from enum import Enum

import torch
import esm
import numpy as np
import pandas as pd
import sklearn
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt



class model_option(Enum):

    esm1b_t33_650M_UR50S = 'esm1b_t33_650M_UR50S'
    esm1_t34_670M_UR50S = 'esm1_t34_670M_UR50S'
    esm1_t34_670M_UR50D = 'esm1_t34_670M_UR50D'
    esm1_t34_670M_UR100 = 'esm1_t34_670M_UR100'
    esm1_t12_85M_UR50S = 'esm1_t12_85M_UR50S'
    esm1_t6_43M_UR50S = 'esm1_t6_43M_UR50S'



@large_task
def pred_reps(
    data_dir: LatchFile,
    output_dir: Optional[str],
    protein_name: str,
    model_arch: model_option = model_option.esm1_t6_43M_UR50S,
    perplexity: int = 30,
    x_axis: str = 'tsne dim. 1',
    y_axis: str = 'tsne dim. 2',
    cbar_title: str = 'fitness',
) -> (int):

    # set output file
    if not output_dir:
        output_path = Path("").resolve()

    else:
        output_path = output_dir

    import esm

    # load ESM model variant

    # ESM-1B
    if model_arch.value == 'esm1b_t33_650M_UR50S':
        # layers = 33, parameter_size = 650M
        model, alphabet = esm.pretrained.esm1b_t33_650M_UR50S()
        final_layer = 33 # final layer output

    elif model_arch.value == 'esm1_t34_670M_UR50S':
        # layers = 34, parameter_size = 670M
        model, alphabet = esm.pretrained.esm1_t34_670M_UR50S()
        final_layer = 34

    elif model_arch.value == 'esm1_t34_670M_UR50D':
        # layers = 34, parameter_size = 670M
        model, alphabet = esm.pretrained.esm1_t34_670M_UR50D()
        final_layer = 34 # final layer output

    elif model_arch.value == 'esm1_t34_670M_UR100':
        # layers = 34, parameter_size = 670M
        model, alphabet = esm.pretrained.esm1_t34_670M_UR100()
        final_layer = 34

    elif model_arch.value == 'esm1_t12_85M_UR50S':
        # layers = 12, parameter_size = 85M
        model, alphabet = esm.pretrained.esm1_t12_85M_UR50S()
        final_layer = 12

    elif model_arch.value == 'esm1_t6_43M_UR50S':
        # layers = 6, parameter size = 43M
        model, alphabet = esm.pretrained.esm1_t6_43M_UR50S()
        final_layer = 6

    else:
        # incorrect model architecture option
        print(f'Error: your choosen value is {model_arch}, but only values between 0-5 are acceptable...')
        quit()
   
    # convert from train to eval mode.
#   model.eval()

 #   batch_converter = alphabet.get_batch_converter()
 
    # load data
   # data = pd.read_csv(data_dir)

    # convert data to tuple
   # data_tuple = [(header, seq) for header, seq in zip(data.iloc[:,0], data.iloc[:,1])]

    # convert protein sequences into readable token sequences
   # batch_labels, batch_strs, batch_tokens = batch_converter(data_tuple)


    # extract per-residue representations
   # with torch.no_grad():
   #     token_representations = model(batch_tokens, repr_layers = [final_layer], return_contacts = False)["representations"][final_layer]

    # save sequence presentations as numpy files

    # generate per-sequence representation (average)
    # NOTE: ESM models use token indx 0 as the beginning-of-sequence token.
   # sequence_representations = []
   # sequence_representations = np.zeros((len(data), token_representations.shape[2]))

   # for i, (_, seq) in enumerate(data_tuple):
   #     sequence_representations[i, :] = token_representations[i,1:len(seq)+1].mean(0).numpy()
    
    # save mean sequence representations as numpy files

    # create dataframe that contains ESM representations
   # rep_columns = [f'representation_{ii}' for ii in range(sequence_representations.shape[1])]
   # ESM_rep_df = pd.DataFrame(sequence_representations, columns = rep_columns)
   # final_df = pd.concat((data, ESM_rep_df), axis = 1)
 
    # save csv file
   # final_df.to_csv(f"/root/protein_seq_mean_reprs.csv", index = False)

    # convert to array
   # protein_reps = final_df.iloc[:,3:].values

   # rep_embed = TSNE(n_components = 2, init = 'random', perplexity = perplexity).fit_transform(protein_reps)

    # convert to dataframe
   # rep_embed_df = pd.DataFrame(rep_embed, columns = ['tsne_0', 'tsne_1'])
   # embed_final_df = pd.concat((data, rep_embed_df), axis = 1)
   # embed_final_df.to_csv(f"/root/tsne_seq_mean_reprs.csv", index = False)

    # plot and save fig
   # plt.figure(dpi = 300)
   # plt.scatter(rep_embed[:, 0], rep_embed[:,1], c = data.iloc[:,2].values)
   # cbar = plt.colorbar()
   # cbar.set_label(cbar_title, rotation = 270, labelpad = 15)
   # plt.xlabel(x_axis)
   # plt.ylabel(y_axis)
    
   # plt.savefig(f"/root/plot_seq_mean_reprs.png")

 
 #   return (LatchFile("/root/protein_seq_mean_reprs.csv", f"latch:///{output_path}{protein_name}_mean_seq_reprs.csv"),
 #           LatchFile("/root/tsne_seq_mean_reprs.csv", f"latch:///{output_path}tsne_{protein_name}_mean_seq_reprs.csv"),
 #           LatchFile("/root/plot_seq_mean_reprs.png", f"latch:///{output_path}plot_{protein_name}_mean_seq_reprs.png"),
 #   )

    return perplexity

@workflow
def fair_esm(
    data_dir: LatchFile,
    output_dir: Optional[str],
    protein_name: str,
    model_arch: model_option = model_option.esm1_t6_43M_UR50S,
    perplexity: int = 30,
    x_axis: str = 'tsne dim. 1',
    y_axis: str = 'tsne dim. 2',
    cbar_title: str = 'fitness',
) -> int:

    """Evolutionary Scale Modeling (ESM) embeds biologically meaningful representations for protein sequences using pretrained Transform protein language models from Facebook AI research.

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


    ----


    __metadata__:
        display_name: ESM
        author:
            name: FAIR
            email:
            github: [facebookresearch/esm](https://github.com/facebookresearch/esm)
        repository:
        liscense:
             id: MIT

    Args:

        data_dir:
          Input data of interest for predicting protein representations.

          __metadata__:
            display_name: Protein data csv Dir

        output_dir:
          Output path of the predicted protein representations.

          __metadata__:
            display_name: Output directory containing the protein representations in a .csv format
       
        protein_name:
          Name of the protein of interest.

          __metadata__:
            display_name: What is the name of the protein of interest?

        model_arch:
          Model option.

          __metadata__:
            display_name: Pick pretrained model.
 
        perplexity:
          A hyperparameter related to the number of nearest neighbors used in manifold learning algorithms.

          __metadata__:
            display_name: Pick hyperparameter value (only positive integer values).

        x_axis:
          X-axis label for the low-dimensional plot

          __metadata__:
            display_name: Pick a title for the x-axis of the tsne plot.
 
        y_axis:
          Y-axis label for the low-dimensional plot

          __metadata__:
            display_name: Pick a title for the y-axis of the tsne plot.
 
        cbar_title:
          Colorbar title for the low-dimensional plot

          __metadata__:
            display_name: Pick a title for the colorbar.
    """



    return pred_reps(
                data_dir=data_dir,
                output_dir=output_dir,
                protein_name=protein_name,
                model_arch=model_arch,
                perplexity=perplexity,
                x_axis=x_axis,
                y_axis=y_axis,
                cbar_title=cbar_title,
)


