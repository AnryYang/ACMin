## Requirements
- Linux
- Python 2.7
- pip install munkres==1.0.12


## Datasets
Download from [here](https://renchi.ac.cn/#datasets)

## Running
```
## Generate clusters
$ python2.7 node_cluster.py --data cora 
```

## Evaluation
```
## CA/NMI scores
$ python2.7 eval_ca_nmi.py --data cora --cfile sc.cora.7.cluster.txt
## Modularity scores
$ python2.7 eval_mod_ent.py --data cora --cfile sc.cora.7.cluster.txt
## AAMC scores
$ python2.7 eval_aamc.py --data cora --cfile sc.cora.7.cluster.txt
```

## Citation
@article{yang2021effective,
  title={Effective and Scalable Clustering on Massive Attributed Graphs},
  author={Yang, Renchi and Shi, Jieming and Yang, Yin and Huang, Keke and Zhang, Shiqi and Xiao, Xiaokui},
  booktitle = {Proceedings of The Web Conference 2021},
  year={2021},
  publisher={Association for Computing Machinery}
}
```
