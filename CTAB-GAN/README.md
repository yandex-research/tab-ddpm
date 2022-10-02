# CTAB-GAN
This is the official git paper [CTAB-GAN: Effective Table Data Synthesizing](https://proceedings.mlr.press/v157/zhao21a.html). The paper is published on Asian Conference on Machine Learning (ACML 2021), please check our pdf on PMLR website for our newest version of [paper](https://proceedings.mlr.press/v157/zhao21a.html), it adds more content on time consumption analysis of training CTAB-GAN. If you have any question, please contact `z.zhao-8@tudelft.nl` for more information.


## Prerequisite

The required package version
```
numpy==1.21.0
torch==1.9.1
pandas==1.2.4
sklearn==0.24.1
dython==0.6.4.post1
scipy==1.4.1
```

## Example
`Experiment_Script_Adult.ipynb` is an example notebook for training CTAB-GAN with Adult dataset. The dataset is alread under `Real_Datasets` folder.
The evaluation code is also provided.

## For large dataset

If your dataset has large number of column, you may encounter the problem that our currnet code cannot encode all of your data since CTAB-GAN will wrap the encoded data into an image-like format. What you can do is changing the line 341 and 348 in `model/synthesizer/ctabgan_synthesizer.py`. The number in the `slide` list
```
sides = [4, 8, 16, 24, 32]
```
is the side size of image. You can enlarge the list to [4, 8, 16, 24, 32, 64] or [4, 8, 16, 24, 32, 64, 128] for accepting larger dataset.

## Bibtex

To cite this paper, you could use this bibtex

```
@InProceedings{zhao21,
  title = 	 {CTAB-GAN: Effective Table Data Synthesizing},
  author =       {Zhao, Zilong and Kunar, Aditya and Birke, Robert and Chen, Lydia Y.},
  booktitle = 	 {Proceedings of The 13th Asian Conference on Machine Learning},
  pages = 	 {97--112},
  year = 	 {2021},
  editor = 	 {Balasubramanian, Vineeth N. and Tsang, Ivor},
  volume = 	 {157},
  series = 	 {Proceedings of Machine Learning Research},
  month = 	 {17--19 Nov},
  publisher =    {PMLR},
  pdf = 	 {https://proceedings.mlr.press/v157/zhao21a/zhao21a.pdf},
  url = 	 {https://proceedings.mlr.press/v157/zhao21a.html}
}


```
