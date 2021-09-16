# MUGE Image Caption Baseline

## Requirements and Installation
More details see [fairseq](https://github.com/pytorch/fairseq). Briefly,

* python == 3.6.4
* pytorch == 1.7.1

**Installing fairseq**
```bash
git clone https://github.com/MUGE-2021/image-caption-baseline
cd muge_baseline/fairseq
pip install --editable .
```

## Getting Started

### Training && Inference
```bash
bash run_scripts/train_caption.sh
bash run_scripts/generate_caption.sh
```

## Reference
```
@inproceedings{M6,
  author    = {Junyang Lin and
               Rui Men and
               An Yang and
               Chang Zhou and
               Ming Ding and
               Yichang Zhang and
               Peng Wang and
               Ang Wang and
               Le Jiang and
               Xianyan Jia and
               Jie Zhang and
               Jianwei Zhang and
               Xu Zou and
               Zhikang Li and
               Xiaodong Deng and
               Jie Liu and
               Jinbao Xue and
               Huiling Zhou and
               Jianxin Ma and
               Jin Yu and
               Yong Li and
               Wei Lin and
               Jingren Zhou and
               Jie Tang and
               Hongxia Yang},
  title     = {{M6:} {A} Chinese Multimodal Pretrainer},
  year      = {2021},
  booktitle = {Proceedings of the 27th ACM SIGKDD Conference on Knowledge Discovery & Data Mining},
  pages     = {3251–3261},
  numpages  = {11},
  location  = {Virtual Event, Singapore},
}

@article{M6-T,
  author    = {An Yang and
               Junyang Lin and
               Rui Men and
               Chang Zhou and
               Le Jiang and
               Xianyan Jia and
               Ang Wang and
               Jie Zhang and
               Jiamang Wang and
               Yong Li and
               Di Zhang and
               Wei Lin and
               Lin Qu and
               Jingren Zhou and
               Hongxia Yang},
  title     = {{M6-T:} Exploring Sparse Expert Models and Beyond},
  journal   = {CoRR},
  volume    = {abs/2105.15082},
  year      = {2021}
}
```

