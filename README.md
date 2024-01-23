# <img src="logo.png" alt="Logo" width="100" height="45"/>: Target Sequence-Conditioned Generation of Peptide Binders via Masked Language Modeling 
In this work, we introduce **PepMLM**, a purely target sequence-conditioned de novo generator of linear peptide binders. By employing a novel masking strategy that uniquely positions cognate peptide sequences at the terminus of target protein sequences, PepMLM tasks the state-of-the-art ESM-2 pLM to fully reconstruct the binder region, achieving low perplexities matching or improving upon previously-validated peptide-protein sequence pairs. After successful *in silico* benchmarking with AlphaFold-Multimer, we experimentally verify PepMLM's efficacy via fusion of model-derived peptides to E3 ubiquitin ligase domains, demonstrating endogenous degradation of target substrates in cellular models. In total, PepMLM enables the generative design of candidate binders to any target protein, without the requirement of target structure, empowering downstream programmable proteome editing applications.

![Pepmlm Image](pepmlm.png)

Check out our [manuscript](https://arxiv.org/abs/2310.03842) on the *arXiv*!

- HuggingFace: [Link](https://huggingface.co/TianlaiChen/PepMLM-650M)
- Demo: HuggingFace Space Demo [Link](https://huggingface.co/spaces/TianlaiChen/PepMLM).
- Colab Notebook: [Link](https://colab.research.google.com/drive/1u0i-LBog_lvQ5YRKs7QLKh_RtI-tV8qM?usp=sharing)

```
# Load model directly
from transformers import AutoTokenizer, AutoModelForMaskedLM

tokenizer = AutoTokenizer.from_pretrained("TianlaiChen/PepMLM-650M")
model = AutoModelForMaskedLM.from_pretrained("TianlaiChen/PepMLM-650M")
```


# Points to be added in this forked repository
- add wandb integration into train.py
- create generation_example.py where the candidates of binders and the predicted structured are shown up in wandb dashboard
- add environment.yml of esmfold

## how to use 
```
# create environment
conda env create -f environment.yml
conda activate esmfold
pip install "fair-esm[esmfold]"
pip install 'dllogger @ git+https://github.com/NVIDIA/dllogger.git'
pip install 'openfold @ git+https://github.com/aqlaboratory/openfold.git@4b41059694619831a7db195b7e0988fc4ff3a307'


# run training
python3 scripts/train.py --wandb_entity <wandb entity> --wandb_project <wandb project> --train_data ./scripts/data/gen_650M.csv --test_data ./scripts/data/test.csv
# ex: python3 scripts/train.py --wandb_entity wandb-japan --wandb_project pepmlm --train_data ./scripts/data/gen_650M.csv --test_data ./scripts/data/test.csv 

# generate binders
python3 scripts/generation_example.py --wandb_entity <wandb entity> --wandb_project <wandb project> --artifacts_path <fined model's artifact path > --protein_seq <your sequence data>
# ex: python3 scripts/generation_example.py --wandb_entity wandb-japan --wandb_project pepmlm --artifacts_path wandb-japan/pepmlm/model-4xrpjde9:v0 --protein_seq MSGIALSRLAQERKAWRKDHPFGFVAVPTKNPDGTMNLMNWECAIPGKKGTPWEGGLFKLRMLFKDDYPSSPPKCKFEPPLFHPNVYPSGTVCLSILEEDKDWRPAITIKQILLGIQELLNEPNIQDPAQAEAYTIYCQNRVEYEKRVRAQAKKFAPS
```

# License
Shield: [![CC BY-NC-SA 4.0][cc-by-nc-sa-shield]][cc-by-nc-sa]

This work is licensed under a
[Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License][cc-by-nc-sa].

[![CC BY-NC-SA 4.0][cc-by-nc-sa-image]][cc-by-nc-sa]

[cc-by-nc-sa]: http://creativecommons.org/licenses/by-nc-sa/4.0/
[cc-by-nc-sa-image]: https://licensebuttons.net/l/by-nc-sa/4.0/88x31.png
[cc-by-nc-sa-shield]: https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg
Contact: pranam.chatterjee@duke.edu

# Citation
```
@article{chen2023pepmlm,
  title={PepMLM: Target Sequence-Conditioned Generation of Peptide Binders via Masked Language Modeling},
  author={Chen, Tianlai and Pertsemlidis, Sarah and Kavirayuni, Venkata Srikar and Vure, Pranay and Pulugurta, Rishab and Hsu, Ashley and Vincoff, Sophia and Yudistyra, Vivian and Hong, Lauren and Wang, Tian and others},
  journal={ArXiv},
  year={2023},
  publisher={arXiv}
}
```
