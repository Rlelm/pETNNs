## pETNNs: Partial Evolutionary Tensor Neural Networks for Solving Time-dependent Partial Differential Equations
by Tunan Kao, He Zhang, Lei Zhang, Jin Zhao

### Transport equation

- 3D case
```
python etnn.py --dim=3 --scheme='RK4' --num_update=300 --t=1 --dt=5e-3 
```

- 10D case
```
python etnn.py --dim=10 --scheme='RK4' --num_update=600 --t=1 --dt=5e-3
```

### Dependencies
- torch==2.1.1
- numpy==1.25.2


### Citations
```
@misc{kao2024petnnspartialevolutionarytensor,
      title={pETNNs: Partial Evolutionary Tensor Neural Networks for Solving Time-dependent Partial Differential Equations}, 
      author={Tunan Kao and He Zhang and Lei Zhang and Jin Zhao},
      year={2024},
      eprint={2403.06084},
      archivePrefix={arXiv},
      primaryClass={math.NA},
      url={https://arxiv.org/abs/2403.06084}, 
}
```
### License