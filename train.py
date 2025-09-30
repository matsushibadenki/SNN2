# matsushibadenki/snn/train.py
# (旧 snn_research/training/main.py)
#
# 新しい統合学習実行スクリプト
#
# 機能:
# - ロードマップ フェーズ2「2.2. 統合された学習パイプライン」に対応。
# - DIコンテナを使用して、学習に必要なコンポーネント（モデル, データセット,
#   Optimizer, Loss, Trainerなど）を動的に組み立てる。
# - --data_format 引数で、異なる形式のデータセットをシームレスに切り替え可能。
# - --override_config 引数で、コマンドラインから任意の設定を上書きできる。
# - 分散学習 (`--distributed`) にも対応。

import argparse
import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, random_split, DistributedSampler
from dependency_injector.wiring import inject, Provide

from app.containers import TrainingContainer
from snn_research.data.datasets import get_dataset_class, DistillationDataset, DataFormat
from snn_research.training.trainers import BreakthroughTrainer, DistillationTrainer

# DIコンテナのセットアップ
container = TrainingContainer()
# 設定のマージ（ファイル -> コマンドライン）
# ... (main関数内で実行)

@inject
def train(
    args,
    snn_model: BreakthroughSNN = Provide[TrainingContainer.snn_model],
    tokenizer = Provide[TrainingContainer.tokenizer],
    config = Provide[TrainingContainer.config]
):
    """学習プロセスを実行するメイン関数"""
    is_distributed = args.distributed
    rank = int(os.environ.get("LOCAL_RANK", -1))
    
    device = f'cuda:{rank}' if is_distributed else get_auto_device()
    snn_model.to(device)

    # --- データセットとデータローダーの準備 ---
    is_distillation = config.training.type() == "distillation"
    if is_distillation:
        dataset = DistillationDataset(
            file_path=os.path.join(args.data_path, "distillation_data.jsonl"),
            data_dir=args.data_path,
            tokenizer=tokenizer,
            max_seq_len=snn_model.time_steps
        )
    else:
        DatasetClass = get_dataset_class(DataFormat(config.data.format()))
        dataset = DatasetClass(
            file_path=args.data_path,
            tokenizer=tokenizer,
            max_seq_len=snn_model.time_steps
        )
        
    train_size = int((1.0 - config.data.split_ratio()) * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_sampler = DistributedSampler(train_dataset) if is_distributed else None
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size(),
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        collate_fn=collate_fn(tokenizer)
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.training.batch_size(),
        shuffle=False,
        collate_fn=collate_fn(tokenizer)
    )

    # --- DIコンテナから学習コンポーネントを取得 ---
    optimizer = container.optimizer(params=snn_model.parameters())
    scheduler = container.scheduler(optimizer=optimizer) if config.training.use_scheduler() else None

    if is_distributed:
        snn_model = DDP(snn_model, device_ids=[rank])
    
    # アストロサイトネットワークを初期化 (オプション)
    astrocyte = container.astrocyte_network(snn_model=snn_model) if args.use_astrocyte else None

    # トレーナーを選択して初期化
    if is_distillation:
        trainer: DistillationTrainer = container.distillation_trainer(
            model=snn_model, optimizer=optimizer, scheduler=scheduler, device=device, rank=rank, astrocyte_network=astrocyte
        )
    else:
        trainer: BreakthroughTrainer = container.standard_trainer(
            model=snn_model, optimizer=optimizer, scheduler=scheduler, device=device, rank=rank, astrocyte_network=astrocyte
        )

    # --- 学習ループの実行 ---
    print(f"🚀 学習を開始します (Device: {device}, Distributed: {is_distributed})")
    start_epoch = 0
    if args.resume_path:
        start_epoch = trainer.load_checkpoint(args.resume_path)

    for epoch in range(start_epoch, config.training.epochs()):
        if is_distributed:
            train_sampler.set_epoch(epoch)
        
        train_metrics = trainer.train_epoch(train_loader, epoch)
        
        if rank in [-1, 0] and (epoch % config.training.eval_interval() == 0 or epoch == config.training.epochs() - 1):
            val_metrics = trainer.evaluate(val_loader, epoch)
            
            if epoch % config.training.log_interval() == 0:
                checkpoint_path = os.path.join(config.training.log_dir(), f"checkpoint_epoch_{epoch}.pth")
                trainer.save_checkpoint(
                    path=checkpoint_path,
                    epoch=epoch,
                    metric_value=val_metrics.get('total', float('inf')),
                    tokenizer_name=config.data.tokenizer_name(),
                    config=config.model.to_dict()
                )

    if rank in [-1, 0]:
        print("✅ 学習が完了しました。")

def collate_fn(tokenizer):
    def collate(batch):
        # バッチ内の最大の長さにパディング
        inputs = [item[0] for item in batch]
        targets = [item[1] for item in batch]
        
        # Distillationの場合、teacher_logitsもパディング
        if len(batch[0]) > 2:
            logits = [item[2] for item in batch]
            padded_logits = torch.nn.utils.rnn.pad_sequence(logits, batch_first=True, padding_value=0)
            
            padded_inputs = torch.nn.utils.rnn.pad_sequence(inputs, batch_first=True, padding_value=tokenizer.pad_token_id)
            padded_targets = torch.nn.utils.rnn.pad_sequence(targets, batch_first=True, padding_value=tokenizer.pad_token_id)

            return padded_inputs, padded_targets, padded_logits
        else:
            padded_inputs = torch.nn.utils.rnn.pad_sequence(inputs, batch_first=True, padding_value=tokenizer.pad_token_id)
            padded_targets = torch.nn.utils.rnn.pad_sequence(targets, batch_first=True, padding_value=tokenizer.pad_token_id)
            return padded_inputs, padded_targets
    return collate


def get_auto_device() -> str:
    if torch.cuda.is_available(): return "cuda"
    if torch.backends.mps.is_available(): return "mps"
    return "cpu"

def main():
    parser = argparse.ArgumentParser(description="SNN 統合学習スクリプト")
    parser.add_argument("--config", type=str, default="configs/base_config.yaml", help="基本設定ファイル")
    parser.add_argument("--model_config", type=str, required=True, help="モデルアーキテクチャ設定ファイル")
    parser.add_argument("--data_path", type=str, help="データセットのパス")
    parser.add_argument("--override_config", type=str, action='append', help="設定を上書き (例: 'training.epochs=5')")
    parser.add_argument("--distributed", action="store_true", help="分散学習を有効にする")
    parser.add_argument("--resume_path", type=str, help="チェックポイントから学習を再開する")
    parser.add_argument("--use_astrocyte", action="store_true", help="アストロサイトネットワークを有効にする")
    
    args = parser.parse_args()

    # DIコンテナの設定をロード
    container.config.from_yaml(args.config)
    container.config.from_yaml(args.model_config)
    if args.data_path:
        container.config.data.path.from_value(args.data_path)
    if args.override_config:
        for override in args.override_config:
            key, value = override.split('=', 1)
            container.config.from_dict({key: value})

    # DDPの初期化
    if args.distributed:
        dist.init_process_group(backend="nccl")

    train(args)

    if args.distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
