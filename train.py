# Copyright (c) 2024-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

# 导入所需的Python标准库
import argparse  # 用于命令行参数解析
import json  # 用于处理JSON数据
import os  # 操作系统功能
import pickle  # Python对象序列化
import random  # 用于生成随机数

import numpy as np  # 用于科学计算
import torch  # PyTorch深度学习框架

# 导入项目自定义模块
import src
from src.envs import ENVS, build_env  # 环境相关函数
from src.evaluator import Evaluator  # 评估器
from src.model import build_modules, check_model_params  # 模型相关函数
from src.slurm import (  # SLURM作业调度器相关工具
    init_distributed_mode,
    init_signal_handler,
)
from src.trainer import Trainer  # 训练器
from src.utils import bool_flag, initialize_exp  # 工具函数

# 设置numpy错误处理为抛出异常，便于调试
np.seterr(all="raise")


def get_parser():
    """
    生成参数解析器。
    定义了训练和评估过程中使用的所有命令行参数。
    """
    # 解析参数
    parser = argparse.ArgumentParser(description="Language transfer")

    # 主要参数
    parser.add_argument(
        "--dump_path",
        type=str,
        default="/path/to/your/storage",
        help="Experiment dump path",
    )  # 实验结果保存路径
    parser.add_argument(
        "--exp_name", type=str, default="debug", help="Experiment name"
    )  # 实验名称
    parser.add_argument(
        "--save_periodic",
        type=int,
        default=0,
        help="Save the model periodically (0 to disable)",
    )  # 定期保存模型（0表示禁用）
    parser.add_argument(
        "--exp_id", type=str, default="", help="Experiment ID"
    )  # 实验ID

    # float16 / AMP API 混合精度训练相关参数
    parser.add_argument(
        "--fp16", type=bool_flag, default=True, help="Run model with float16"
    )  # 是否使用float16精度
    parser.add_argument(
        "--amp",
        type=int,
        default=2,
        help="Use AMP wrapper for float16 / distributed / gradient accumulation. Level of optimization. -1 to disable.",
    )  # 自动混合精度优化级别

    # 模型参数
    parser.add_argument(
        "--emb_dim", type=int, default=640, help="Embedding layer size"
    )  # 嵌入层维度
    parser.add_argument(
        "--n_enc_layers",
        type=int,
        default=6,
        help="Number of Transformer layers in the encoder",
    )  # 编码器中Transformer层数量
    parser.add_argument(
        "--n_dec_layers",
        type=int,
        default=6,
        help="Number of Transformer layers in the decoder",
    )  # 解码器中Transformer层数量
    parser.add_argument(
        "--n_heads", type=int, default=10, help="Number of Transformer heads"
    )  # Transformer注意力头数量
    parser.add_argument(
        "--dropout", type=float, default=0, help="Dropout"
    )  # Dropout比率
    parser.add_argument(
        "--attention_dropout",
        type=float,
        default=0,
        help="Dropout in the attention layer",
    )  # 注意力层中的Dropout
    parser.add_argument(
        "--share_inout_emb",
        type=bool_flag,
        default=True,
        help="Share input and output embeddings",
    )  # 是否共享输入和输出嵌入
    parser.add_argument(
        "--sinusoidal_embeddings",
        type=bool_flag,
        default=False,
        help="Use sinusoidal embeddings",
    )  # 是否使用正弦嵌入
    parser.add_argument(
        "--max_src_len",
        type=int,
        default=0,
        help="force all inputs to be not longer than src_len. 0 means no restrictions",
    )  # 输入长度限制

    # 训练参数
    parser.add_argument(
        "--env_base_seed",
        type=int,
        default=-1,
        help="Base seed for environments (-1 to use timestamp seed)",
    )  # 环境基础随机种子
    parser.add_argument(
        "--max_len", type=int, default=512, help="Maximum sequences length"
    )  # 最大序列长度
    parser.add_argument(
        "--max_output_len",
        type=int,
        default=512,
        help="max length of output, beam max size",
    )  # 最大输出长度，束搜索最大尺寸

    parser.add_argument(
        "--batch_size", type=int, default=32, help="Number of sentences per batch"
    )  # 每批次句子数
    parser.add_argument(
        "--batch_size_eval",
        type=int,
        default=128,
        help="Number of sentences per batch during evaluation",
    )  # 评估时每批次句子数
    parser.add_argument(
        "--eval_size", type=int, default=10000, help="Size of valid and test samples"
    )  # 验证和测试样本数量
    parser.add_argument(
        "--optimizer",
        type=str,
        default="adam,lr=0.0001",
        help="Optimizer (SGD / RMSprop / Adam, etc.)",
    )  # 优化器类型及参数
    parser.add_argument(
        "--clip_grad_norm",
        type=float,
        default=5,
        help="Clip gradients norm (0 to disable)",
    )  # 梯度裁剪范数
    parser.add_argument(
        "--epoch_size",
        type=int,
        default=300000,
        help="Epoch size / evaluation frequency",
    )  # 每轮大小/评估频率
    parser.add_argument(
        "--max_epoch", type=int, default=100000, help="Maximum epoch size"
    )  # 最大训练轮数
    parser.add_argument(
        "--stopping_criterion",
        type=str,
        default="",
        help="Stopping criterion, and number of non-increase before stopping the experiment",
    )  # 停止训练的标准
    parser.add_argument(
        "--validation_metrics", type=str, default="", help="Validation metrics"
    )  # 验证指标
    parser.add_argument(
        "--accumulate_gradients",
        type=int,
        default=1,
        help="Accumulate model gradients over N iterations (N times larger batch sizes)",
    )  # 梯度累积迭代次数
    parser.add_argument(
        "--num_workers",
        type=int,
        default=10,
        help="Number of CPU workers for DataLoader",
    )  # DataLoader的CPU工作线程数

    # 导出数据/重新加载数据
    parser.add_argument(
        "--export_data",
        type=bool_flag,
        default=False,
        help="Export data and disable training.",
    )  # 是否导出数据并禁用训练
    parser.add_argument(
        "--reload_data",
        type=str,
        default="",
        help="Load dataset from the disk (task1,train_path1,valid_path1,test_path1_1,...;task2,train_path2,valid_path2,test_path2_1,...)",
    )  # 从磁盘加载数据集
    parser.add_argument(
        "--reload_size",
        type=int,
        default=-1,
        help="Reloaded training set size (-1 for everything)",
    )  # 重新加载的训练集大小

    # 环境参数
    parser.add_argument(
        "--env_name", type=str, default="ode", help="Environment name"
    )  # 环境名称
    ENVS[parser.parse_known_args()[0].env_name].register_args(
        parser
    )  # 根据环境名称注册特定环境的参数

    # 任务
    parser.add_argument(
        "--tasks", type=str, default="ode_lyapunov", help="Tasks"
    )  # 任务名称，默认为李雅普诺夫ODE任务

    # 束搜索配置
    parser.add_argument(
        "--beam_eval",
        type=bool_flag,
        default=False,
        help="Evaluate with beam search decoding.",
    )  # 是否使用束搜索解码进行评估
    parser.add_argument(
        "--beam_size",
        type=int,
        default=1,
        help="Beam size, default = 1 (greedy decoding)",
    )  # 束大小，默认为1（贪婪解码）
    parser.add_argument(
        "--beam_length_penalty",
        type=float,
        default=1,
        help="Length penalty, values < 1.0 favor shorter sentences, while values > 1.0 favor longer ones.",
    )  # 长度惩罚，<1.0偏好短句，>1.0偏好长句
    parser.add_argument(
        "--beam_early_stopping",
        type=bool_flag,
        default=True,
        help="Early stopping, stop as soon as we have `beam_size` hypotheses, although longer ones may have better scores.",
    )  # 提前停止，一旦有beam_size个假设就停止

    # 重新加载预训练模型/检查点
    parser.add_argument(
        "--reload_model", type=str, default="", help="Reload a pretrained model"
    )  # 重新加载预训练模型
    parser.add_argument(
        "--reload_checkpoint", type=str, default="", help="Reload a checkpoint"
    )  # 重新加载检查点

    # 评估
    parser.add_argument(
        "--eval_only", type=bool_flag, default=False, help="Only run evaluations"
    )  # 仅运行评估
    parser.add_argument(
        "--eval_from_exp", type=str, default="", help="Path of experiment to use"
    )  # 要使用的实验路径
    parser.add_argument(
        "--eval_data", type=str, default="", help="Path of data to eval"
    )  # 要评估的数据路径
    parser.add_argument(
        "--eval_verbose", type=int, default=0, help="Export evaluation details"
    )  # 导出评估详情
    parser.add_argument(
        "--eval_verbose_print",
        type=bool_flag,
        default=False,
        help="Print evaluation details",
    )  # 打印评估详情

    # 调试
    parser.add_argument(
        "--debug_slurm",
        type=bool_flag,
        default=False,
        help="Debug multi-GPU / multi-node within a SLURM job",
    )  # 在SLURM作业中调试多GPU/多节点
    parser.add_argument(
        "--debug", help="Enable all debug flags", action="store_true"
    )  # 启用所有调试标志

    # CPU/多GPU/多节点
    parser.add_argument(
        "--cpu", type=bool_flag, default=False, help="Run on CPU"
    )  # 在CPU上运行
    parser.add_argument(
        "--local_rank", type=int, default=-1, help="Multi-GPU - Local rank"
    )  # 多GPU-本地排名
    parser.add_argument(
        "--master_port",
        type=int,
        default=-1,
        help="Master port (for multi-node SLURM jobs)",
    )  # 主端口（用于多节点SLURM作业）

    return parser


def main(params):
    """
    主函数：执行实验的训练和评估流程

    参数:
        params: 解析得到的命令行参数
    """
    # 初始化多GPU/多节点训练
    # 初始化实验/SLURM信号处理器（用于时间限制/抢占）
    init_distributed_mode(params)  # 初始化分布式训练模式
    logger = initialize_exp(params)  # 初始化实验日志记录器
    init_signal_handler()  # 初始化信号处理器

    # 设置CPU/CUDA
    if params.cpu:
        assert not params.multi_gpu  # CPU模式下不能使用多GPU
    else:
        assert torch.cuda.is_available()  # 非CPU模式需要CUDA可用
    src.utils.CUDA = not params.cpu  # 设置全局CUDA标志

    # 构建环境/模块/训练器/评估器
    env = build_env(params)  # 构建实验环境
    modules = build_modules(env, params)  # 构建模型模块
    trainer = Trainer(modules, env, params)  # 创建训练器
    evaluator = Evaluator(trainer)  # 创建评估器

    # 评估模式
    if params.eval_only:
        scores = evaluator.run_all_evals()  # 运行所有评估
        for k, v in scores.items():
            logger.info("%s -> %.6f" % (k, v))  # 记录评估结果
        logger.info("__log__:%s" % json.dumps(scores))  # 以JSON格式记录完整评估结果
        exit()  # 评估完成后退出

    # 训练流程
    for _ in range(params.max_epoch):  # 循环进行最大轮次的训练
        logger.info(
            "============ Starting epoch %i ... ============" % trainer.epoch
        )  # 记录开始新的训练轮次

        trainer.n_equations = 0  # 重置方程计数器

        while trainer.n_equations < trainer.epoch_size:  # 当前轮次未完成
            # 训练步骤
            for task_id in np.random.permutation(len(params.tasks)):  # 随机打乱任务顺序
                task = params.tasks[task_id]  # 获取当前任务
                if params.export_data:
                    trainer.export_data(task)  # 导出数据
                else:
                    trainer.enc_dec_step(task)  # 执行编码器-解码器训练步骤
                trainer.iter()  # 更新迭代计数

        logger.info(
            "============ End of epoch %i ============" % trainer.epoch
        )  # 记录轮次结束

        # 评估模型性能
        if params.is_master:  # 仅在主进程中进行评估
            scores = evaluator.run_all_evals()  # 运行所有评估
            logger.info(scores)  # 记录评估分数

            # 打印/JSON日志
            for k, v in scores.items():
                logger.info("%s -> %.6f" % (k, v))  # 记录每项评估指标
            logger.info("__log__:%s" % json.dumps(scores))  # 以JSON格式记录完整评估结果

            # 轮次结束处理
            trainer.save_best_model(scores)  # 保存最佳模型
            trainer.save_periodic()  # 定期保存模型
            trainer.end_epoch(scores)  # 结束当前轮次，更新训练状态


if __name__ == "__main__":
    """
    程序入口点，处理命令行参数并启动实验
    """
    # 生成解析器/解析参数
    parser = get_parser()  # 获取参数解析器
    params = parser.parse_args()  # 解析命令行参数

    # 处理仅评估模式且指定了实验路径的情况
    if params.eval_only and params.eval_from_exp != "":
        # 从pickle文件读取参数
        pickle_file = params.eval_from_exp + "/params.pkl"  # 参数文件路径
        exp_str = params.eval_from_exp  # 实验路径
        assert os.path.isfile(pickle_file)  # 确保参数文件存在
        pk = pickle.load(open(pickle_file, "rb"))  # 加载参数
        pickled_args = pk.__dict__  # 获取参数字典

        # 删除不需要从之前实验加载的参数
        del pickled_args["exp_id"]
        del pickled_args["dump_path"]
        del pickled_args["exp_name"]
        for key in [
            "eval_data",
            "batch_size_eval",
            "local_rank",
            "beam_eval",
            "beam_size",
            "eval_verbose",
            "eval_verbose_print",
            "master_port",
            "max_len",
            "max_output_len",
            "lyap_SOS_checker",
        ]:
            if key in pickled_args:
                del pickled_args[key]

        # 用加载的参数更新当前参数
        for p in params.__dict__:
            if p in pickled_args:
                params.__dict__[p] = pickled_args[p]  # 使用保存的参数覆盖当前参数

        # 设置评估参数
        params.eval_only = True  # 确保仅评估模式
        params.reload_model = (
            exp_str + "/best-" + params.validation_metrics + ".pth"
        )  # 加载最佳模型
        assert os.path.isfile(params.reload_model)  # 确保模型文件存在

        # 如果指定了评估数据，更新相关参数
        if params.eval_data != "":
            params.eval_size = None  # 不限制评估大小
            params.reload_data = (
                params.tasks + "," + params.eval_data + "," + params.eval_data
            )  # 设置数据路径

    # 调试模式设置
    if params.debug:
        params.exp_name = "debug"  # 设置实验名为debug
        if params.exp_id == "":
            params.exp_id = "debug_%08i" % random.randint(
                0, 100000000
            )  # 生成随机调试实验ID
        params.debug_slurm = True  # 启用SLURM调试

    # 检查模型参数
    check_model_params(params)  # 验证模型参数的有效性

    # 运行实验
    main(params)  # 启动主函数执行实验
