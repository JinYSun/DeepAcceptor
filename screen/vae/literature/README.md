# Log
- started with the basic VAE from the selfies repository, trained it on QM9_selected and got good results (https://wandb.ai/rmeinl/selfies-vae/runs/1366bbk9/overview?workspace=user-rmeinl); saved model in /saved_models/QM9
- applied the same model to the ZINC15 dataset (reduced it to the same length of ~65k for train and valid set) and wasn't able to reproduce the good results (https://wandb.ai/rmeinl/selfies-vae/runs/2twvsnoz?workspace=user-rmeinl)
- hypothesis: the GRU model is unable to model longer sequences (mols in the zinc dataset are up to ~3x longer)
- implemented the MDD loss after reading a blog post (https://ermongroup.github.io/blog/a-tutorial-on-mmd-variational-autoencoders/); re-ran the same QM9 model using the MMD loss (https://wandb.ai/rmeinl/selfies-vae/runs/1mbe9fzm?workspace=user-rmeinl)
- experiment: use LSTM as a decoder
- experiment: implement minGPT with a couple of layers as a decoder and see how it performs against the LSTM (try GPT for now and if it works well and I want to scale up I can try the T5 implementation or GPT-2, though I think a small GPT should do it)
- think about: selfies always creates a different alphabet, how would that work with pretraining GPT?

# Wandb
Selfies-vae: https://wandb.ai/rmeinl/selfies-vae?workspace=user-rmeinl
Mol-vae: https://wandb.ai/rmeinl/mol-vae?workspace=user-rmeinl
Selfies-mingpt: https://wandb.ai/rmeinl/selfies-mingpt?workspace=user-rmeinl

# Literature Review

## Datasets to show that the AE learned molecular properties
- Water solubility: [Delayney](https://deepchem.readthedocs.io/en/latest/api_reference/moleculenet.html#delaney-datasets)
- Hydration free energies: [Freesolv](https://deepchem.readthedocs.io/en/latest/api_reference/moleculenet.html#freesolv-dataset)
- Thermodynamic solubility: 
    - [HPPB](https://deepchem.readthedocs.io/en/latest/api_reference/moleculenet.html#hppb-datasets)
    - [Thermosol](https://deepchem.readthedocs.io/en/latest/api_reference/moleculenet.html#thermosol-datasets)
- Liphophilicity: [LIPO](https://deepchem.readthedocs.io/en/latest/api_reference/moleculenet.html#lipo-datasets)
- Free solvation: [SAMPL](https://deepchem.readthedocs.io/en/latest/api_reference/moleculenet.html#sampl-datasets)

## Early Visual Concept Learning with Unsupervised Deep Learning
Link: https://arxiv.org/pdf/1606.05579.pdf
Summary: https://rylanschaeffer.github.io/content/research/early_visual_concept_learning/main.html
- Learning disentangled latent variables

## Generating Sentences from a Continuous Space
Link: https://arxiv.org/pdf/1511.06349.pdf
- Training RNN VAEs

## A Hierarchical Latent Vector Model for Learning Long-Term Structure in Music
Link: https://arxiv.org/pdf/1803.05428.pdf
Github: https://github.com/magenta/magenta/tree/main/magenta/models/music_vae
- VAEs for longer sequences

## A Tutorial on Information Maximizing Variational Autoencoders (InfoVAE)
Link: https://ermongroup.github.io/blog/a-tutorial-on-mmd-variational-autoencoders/
- using an MMD loss instead of the ELBO loss

## Transformer-VAE
Link: https://github.com/Fraser-Greenlee/transformer-vae
- using T5 as an encoder and decoder

## Optimus - VAE with BERT and GPT-2
Link: http://chunyuan.li/papers/Optimus_2020.pdf
Github: https://github.com/ChunyuanLI/Optimus
- Microsoft paper where they used BERT as an encoder and GPT-2 as a decoder with a VAE in the middle

## SINGLE-CELL DATA ANALYSIS USING MMD VARIATIONAL AUTOENCODER
Link: https://www.biorxiv.org/content/10.1101/613414v1.full.pdf
- VAE with MMD on single cell data

## Self-Referencing Embedded Strings (SELFIES): A 100% robust molecular string representation
Github: https://github.com/aspuru-guzik-group/selfies

## TextRNNVAE
Github: https://github.com/chrisvdweth/ml-toolkit/blob/master/pytorch/models/text/autoencoder/textrnnvae.py
- Pytorch guy posted his textrnnvae implementation which I used in mol-vae (molgen)

## minGPT
Github: https://github.com/karpathy/minGPT

## LigGPT: Molecular Generation using a Transformer-Decoder Model
Link: https://chemrxiv.org/engage/api-gateway/chemrxiv/assets/orp/resource/item/60c7588e469df48597f456ae/original/lig-gpt-molecular-generation-using-a-transformer-decoder-model.pdf
Github: https://github.com/devalab/molgpt

# Tackling the problems of longer sequences
## WaveNet
Link: https://arxiv.org/pdf/1609.03499.pdf
- uses CNNs

# Forcing meaning into the latent space
## MIDI-VAE: MODELING DYNAMICS AND INSTRUMENTATION OF MUSIC WITH APPLICATIONS TO STYLE TRANSFER
Link: https://arxiv.org/pdf/1809.07600.pdf
- adds a style classifier to force the 1st dimension of z to model style

