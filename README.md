# PyTorch 论文实现："Global Lyapunov functions: a long-standing open problem in mathematics, with symbolic transformers" (NeurIPS 2024)

## 环境配置
- 环境要求包含在 Lyapunov.yml 中，您可以通过以下命令设置 conda 环境：
```bash
conda env create -f Lyapunov_python_3_10.yml
```
或者，如果您更喜欢或与 Lyapunov_python_3_10 有冲突，您可以使用：
```bash
conda env create -f Lyapunov_python_3_9.yml
```
然后运行：
```bash
conda activate Lyapunov
```

通常，我们在本地使用 Lyapunov_python_3_9（conda 4.12.0 和 pip 24.3.1），在集群上使用 Lyapunov_python_3_10 以解决兼容性问题。它们之间的唯一区别是 python、dreal 和 torch 的版本。这些环境专为 Linux 和 Mac 设计。Windows 用户可能会遇到一些困难，因为 dReal 作为包可能不受支持。

## 基准测试
- 我们在 `/benchmarks` 文件夹中提供了 4 个文件。这些基准代表了引用的 BPoly、FBarr、FLyap 和 FSOSTOOL。

## 数据生成
- 我们提供了三个不同的 json 文件和 `train.py` 文件来生成向后多项式、向后非多项式和前向多项式数据集。请修改 `dump_path` 字段。
- 生成前向多项式数据集时，请分配足够的 CPU 资源以加速生成。此外，如果您想解决障碍问题，请将 `lyap_proper_fwd` 标志更改为 `False`。
- 在 data.prefix 中，您将找到生成的数据集。存储位置也会在 .stderr 文件中显示，位于 "The experiment will be stored in" 之后。

## 数据集创建
- 首先，通过使用带有 `--export_data true` 的 `train.py` 生成示例，例如（您可以指定更多标志来定制生成）：
```bash
python train.py --dump_path /Your/Path/to/storage/ --export_data true --cpu true --reload_data '' --env_base_seed -1  --num_workers 20
```

- 生成足够的样本后，您应该运行 `create_dataset.py` 脚本来清理和合并多个数据集或清理单个数据集。这是训练阶段前的必要步骤。在 `create_dataset.py` 文件中修改路径到您的位置，然后运行：
```bash
python create_dataset.py
```
- 该脚本将创建 3 个重要文件，分别以 .train、.cleaned.valid 和 .cleaned.test 结尾。

## 训练
- 我们提供了一个 json 文件和 `train.py` 文件来启动训练。在 `reload_data` 字符串中，请提供任务名称（始终为 ode_lyapunov）、训练数据集、评估数据集和不同的基准（例如 `"ode_lyapunov,/path/to/your/dataset.train,/path/to/your/dataset.valid.final,benchmarks/BPoly,benchmarks/FBarr,benchmarks/FLyap,benchmarks/FSOSTOOL"`）。

- 您可以使用这些标志来定制生成/训练。例如：
```bash
python train.py

--dump_path "/your/dump/path" #数据集和模型的导出路径
--n_enc_layers 6 #编码器层数
--n_dec_layers 6 #解码器层数
--emb_dim 640 #嵌入维度
--n_heads 10 #注意力头数量
--batch_size 4 #批量大小
--batch_size_eval 16 #评估时的批量大小

--max_src_len 0
--max_len 1024 #序列的最大长度
--max_output_len 512 #输出序列的最大长度
--optimizer "adam_inverse_sqrt,warmup_updates=10000,lr=0.0001"
--epoch_size 300000
--max_epoch 100000
--num_workers 1
--export_data false #生成时设为true，训练时设为false

--eval_size 200
--eval_verbose 0
--beam_eval true

--lyap_polynomial_H true #
--lyap_basic_functions_num true
--lyap_pure_polynomial true #仅多项式系统
--lyap_SOS_checker true #使用平方和检查器进行评估
--lyap_SOS_fwd_gen false #使用平方和检查器生成前向分布示例时设为true

--stopping_criterion "valid_ode_lyapunov_beam_acc,100"
--validation_metrics "valid_ode_lyapunov_beam_acc"

--reload_size -1
--reload_data "ode_lyapunov,/path/to/your/dataset.train,/path/to/youdataset.valid.final,benchmarks/BPoly,benchmarks/FBarr,benchmarks/FLyap,benchmarks/FSOSTOOL" #任务，然后是训练数据集、有效数据集和测试数据集
```
- 环境的完整标志列表以及它们代表的更多详细信息可以在 `ode.py` 的 `register_args` 方法中找到。模型架构和训练参数的完整标志列表可以在 `train.py` 的 `get_parser` 方法中找到。

## 参考文献
此代码在创意共享许可下发布，有关更多详细信息，请参阅 LICENSE 文件。如果您使用此代码，请考虑引用：

```  
@article{alfarano2024global,
  title={Global Lyapunov functions: a long-standing open problem in mathematics, with symbolic transformers},
  author={Alfarano, Alberto and Charton, Fran{\c{c}}ois and Hayat, Amaury},
  journal={arXiv preprint arXiv:2410.08304},
  year={2024}
}
```