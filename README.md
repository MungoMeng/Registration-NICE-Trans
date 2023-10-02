# NICE-Trans: Non-iterative Coarse-to-fine Transformer Networks for Joint Affine and Deformable Image Registration
Recently, Non-Iterative Coarse-to-finE (NICE) registration methods have been proposed to perform coarse-to-fine registration in a single network and showed advantages in both registration accuracy and runtime. However, existing NICE registration methods mainly focus on deformable registration, while affine registration, a common prerequisite, is still reliant on time-consuming traditional optimization-based methods or extra affine registration networks. In addition, existing NICE registration methods are limited by the intrinsic locality of convolution operations. Transformers may address this limitation for their capabilities to capture long-range dependency, but the benefits of using transformers for NICE registration have not been explored. In this study, we propose a Non-Iterative Coarse-to-finE Transformer network (NICE-Trans) for image registration. Our NICE-Trans is the first deep registration method that (i) performs joint affine and deformable coarse-to-fine registration within a single network, and (ii) embeds transformers into a NICE registration framework to model long-range relevance between images. Extensive experiments with seven public datasets show that our NICE-Trans outperforms state-of-the-art registration methods on both registration accuracy and runtime.  
**For more details, please refer to our paper. [[Springer](https://link.springer.com/chapter/10.1007/978-3-031-43999-5_71)] [[arXiv](https://arxiv.org/abs/2307.03421)]**

## Architecture
![architecture](https://github.com/MungoMeng/Registration-NICE-Trans/blob/master/Figure/architecture.png)

## Publication
If this repository helps your work, please kindly cite our paper:
* **Mingyuan Meng, Lei Bi, Michael Fulham, Dagan Feng, Jinman Kim, "Non-iterative Coarse-to-fine Transformer Networks for Joint Affine and Deformable Image Registration," International Conference on Medical Image Computing and Computer Assisted Intervention (MICCAI), pp.750-760, 2023, doi: 10.1007/978-3-031-43999-5_71. [[Springer](https://link.springer.com/chapter/10.1007/978-3-031-43999-5_71)] [[arXiv](https://arxiv.org/abs/2307.03421)]**
