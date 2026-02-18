# ai-model-dev-demos
聚焦大模型训练核心技术的练手仓库，包含分布式并行、DiT/VAE 模型实现、CUDA 算子优化及 LeetCode 并行解法，主打极简可运行、原理可视化。
```
ai-model-dev-demos/
├── distributed-training/  # 分布式训练核心demo
│   ├── tensor_parallel.py  # 张量并行基础实现（以线性层为例）
│   ├── sequence_parallel.py  # 序列并行核心逻辑（token维度拆分）
│   ├── hybrid_parallel_attention.py  # 自注意力混合并行（张量+序列）
│   └── README.md  # 并行策略对比、通信开销分析
├── model-implementations/  # 经典模型极简实现
│   ├── DiT/  # Denoising Diffusion Implicit Models
│   │   ├── mini_dit.py  # 轻量版DiT（MNIST数据集）
│   │   ├── sampling.py  # 推理采样脚本
│   │   └── notes.md  # DiT核心降噪流程拆解
│   └── VAE/
│       ├── mini_vae.py  # Conv+Linear基础VAE
│       ├── latent_vis.py  # 隐空间分布可视化
│       └── loss_analysis.py  # 重构损失+KL散度计算过程
├── cuda-programming/  # CUDA算子开发与优化
│   ├── attention_cuda_kernel.cu  # 自注意力CUDA核函数
│   ├── compile.sh  # Windows/Linux编译脚本
│   ├── benchmark.py  # CPU vs CUDA性能对比
│   └── notes.md  # CUDA优化技巧（共享内存、线程块划分）
├── leetcode-parallel-solutions/  # LeetCode题CUDA并行解法
│   ├── two_sum_cuda.cu  # 两数之和并行实现
│   ├── matrix_mult_cuda.cu  # 矩阵乘法并行优化
│   └── README.md  # 每道题的并行思路+性能对比
├── .gitignore  # 忽略模型权重、编译产物、日志、虚拟环境
└── README.md  # 仓库总说明（当前文件）
```
