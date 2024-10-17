# nelora-lite

**Archived, moved to github repo [NeLoRa_Dataset](https://github.com/daibiaoxuwu/NeLoRa_Dataset)**

A lite version of the original repo [NELoRa-Sensys](https://github.com/hanqingguo/NELoRa-Sensys/)
reproducing the experiments in the SenSys '21 paper "[NELoRa: Towards Ultra-low SNR LoRa Communication with Neural-enhanced Demodulation](https://cse.msu.edu/~caozc/papers/sensys21-li.pdf)".

Now for both nelora and baseline train/test, no need for a separate stage of data-generation (adding artificial noise). 
Noise is added on-the-fly.
This reduces overfitting issues and removes the need for additional harddisk space, also speeding up the process drastically.

Code is kept minimum. Other usability issues, like data balancing and testing during training, are not addressed.
Parameters are hardcoded.

Datasets is based on this [Github Repo](https://github.com/daibiaoxuwu/NeLoRa_Dataset) from [NELoRa-Bench: A Benchmark for Neural-enhanced LoRa Demodulation](https://doi.org/10.48550/arXiv.2305.01573)

Baseline methods uses LoRaPhy from [From Demodulation to Decoding: Toward Complete LoRa PHY Understanding and Implementation](https://doi.org/10.1145/3546869)

please consider to cite our paper if you use the code or data in your research project.
```bibtex
  @inproceedings{nelora2021sensys,
  	title={{NELoRa: Towards Ultra-low SNR LoRa Communication with Neural-enhanced Demodulation}},
  	author={Li, Chenning and Guo, Hanqing and Tong, Shuai and Zeng, Xiao and Cao, Zhichao and Zhang, Mi and Yan, Qiben and Xiao, Li and Wang, Jiliang and Liu, Yunhao},
    	booktitle={In Proceeding of ACM SenSys},
    	year={2021}
  }
```
