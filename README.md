GAATNet
==========

This repo hosts the code and dataset for the paper "Graph Attention-based Adaptive Transfer Learning for Link Prediction", ACM Transactions on Intelligent Systems and Technology, Huashen Lu, Wensheng Gan*, Guoting Chen, Zhichao Huang, Philip S.Yu, pp 1-26, 2026.

## Training

To run:
```
python fine-tune.py --dataset Facebook --epochs 200 --dropout 0.3 --lr 0.01 --Beta 0.3
```

## Citation

If this article or code useful for your project, please refer to

```bibtex
@article{
  title={Graph Attention-based Adaptive Transfer Learning for Link Prediction},
  author={Huashen Lu, Wensheng Gan*, Guoting Chen, Zhichao Huang, and Philip S. Yu},
  journal={ACM Transactions on Intelligent Systems and Technology},
  pages={1-16},
  year={2026},
}
```