# /train_planner.py
# 学習可能プランナーSNN (PlannerSNN) を学習させるためのスクリプト

import argparse
import os
import torch
from dependency_injector.wiring import inject, Provide
from torch.utils.data import DataLoader, Dataset
from typing import List, Dict

from app.containers import TrainingContainer
from snn_research.training.trainers import PlannerTrainer

# DIコンテナのセットアップ
container = TrainingContainer()

class PlannerDataset(Dataset):
    """プランナー学習用のダミーデータセット"""
    def __init__(self, tokenizer, skill_to_id, max_len):
        self.tokenizer = tokenizer
        self.skill_to_id = skill_to_id
        self.max_len = max_len
        
        # (例) 「要約」してから「感情分析」するタスク
        self.data = [
            {
                "request": "この文章を要約して、内容の感情を分析してください。",
                "plan": ["文章要約", "感情分析"]
            },
            {
                "request": "気分はどう？この記事を分析して要約して。",
                "plan": ["感情分析", "文章要約"]
            }
        ]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        request = item["request"]
        plan = item["plan"]
        
        input_ids = self.tokenizer.encode(request, return_tensors='pt', max_length=self.max_len, padding='max_length', truncation=True).squeeze(0)
        
        target_ids = torch.tensor([self.skill_to_id.get(skill, -100) for skill in plan], dtype=torch.long)
        # ターゲットも固定長にする (簡単のため2スキルまでと仮定)
        padded_target = torch.full((2,), -100, dtype=torch.long)
        padded_target[:len(target_ids)] = target_ids
        
        return input_ids, padded_target

@inject
def main(
    config=Provide[TrainingContainer.config],
    tokenizer=Provide[TrainingContainer.tokenizer]
):
    """プランナーの学習を実行するメイン関数"""
    print("🚀 プランナーSNNの学習を開始します...")

    # 簡単なスキル辞書を作成
    # 本来はModelRegistryから動的に取得する
    skills = ["文章要約", "感情分析"]
    skill_to_id = {skill: i for i, skill in enumerate(skills)}

    dataset = PlannerDataset(tokenizer, skill_to_id, config.model.time_steps())
    dataloader = DataLoader(dataset, batch_size=config.training.batch_size())

    device = container.get_auto_device()
    
    # DIコンテナからプランナー用のコンポーネントを取得
    planner_model = container.planner_snn(num_skills=len(skills)).to(device)
    optimizer = container.planner_optimizer(params=planner_model.parameters())
    criterion = container.planner_loss()
    
    trainer = PlannerTrainer(
        model=planner_model,
        optimizer=optimizer,
        criterion=criterion,
        device=device
    )

    for epoch in range(config.training.epochs()):
        trainer.train_epoch(dataloader, epoch)

    # 学習済みモデルの保存
    output_path = "runs/planner_snn.pth"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.save(planner_model.state_dict(), output_path)
    print(f"✅ 学習済みプランナーモデルを '{output_path}' に保存しました。")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="プランナーSNN 学習スクリプト")
    parser.add_argument("--config", type=str, default="configs/base_config.yaml")
    parser.add_argument("--model_config", type=str, default="configs/models/small.yaml")
    args = parser.parse_args()
    
    container.config.from_yaml(args.config)
    container.config.from_yaml(args.model_config)
    container.wire(modules=[__name__])
    
    main()