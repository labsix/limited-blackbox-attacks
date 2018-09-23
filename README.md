# Code for Black-box Adversarial Attacks with Limited Queries and Information
Codebase for reproducing the results in the paper "Black-Box Adversarial Attacks with Limited Queries and Information". The paper can be found [on arxiv](http://arxiv.org/abs/1804.08598), and our explanatory blog post can be found on [labsix.org](http://labsix.org).

To reproduce our results:

1. Make a directory `tools/data`, and in it put the decompressed Inceptionv3 classifier from (http://download.tensorflow.org/models/inception_v3_2016_08_28.tar.gz)
2. Set `IMAGENET_PATH` in main.py, attacks.py, and precompute.py to the location of the ImageNet dataset on your machine.
3. Precompute the starting images (for partial-information and label-only attacks) with `python precompute.py`
4. Run the reproduction scripts with `{query-limited|partial-info|label-only}.sh`, making sure to first edit them specifying an img-index (by default runs on imagenet image 0) 

## Citation
```
@inproceedings{ilyas2018blackbox,
  author = {Andrew Ilyas and Logan Engstrom and Anish Athalye and Jessy Lin},
  title = {Black-box Adversarial Attacks with Limited Queries and Information},
  booktitle = {Proceedings of the 35th International Conference on Machine Learning, {ICML} 2018},
  year = {2018},
  month = jul,
  url = {https://arxiv.org/abs/1804.08598},
}
```
