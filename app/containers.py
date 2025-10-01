# matsushibadenki/snn2/app/containers.py
# DIコンテナの定義ファイル (完全版)
#
# 機能:
# - 勾配ベース学習と生物学的学習の2つのパラダイムをDIコンテナで管理。
# - 設定ファイルの `training.paradigm` の値に応じて、適切なコンポーネント群を構築する。
# - 既存の全機能を維持しつつ、新しい学習方法への拡張性を確保。

import torch
from dependency_injector import containers, providers
from torch.optim import AdamW, Optimizer
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR, LRScheduler
from transformers import AutoTokenizer, AutoModelForCausalLM

# --- プロジェクト内モジュールのインポート (既存) ---
from snn_research.core.snn_core import BreakthroughSNN
from snn_research.deployment import SNNInferenceEngine
from snn_research.training.losses import CombinedLoss, DistillationLoss, SelfSupervisedLoss
from snn_research.training.trainers import BreakthroughTrainer, DistillationTrainer, SelfSupervisedTrainer
from snn_research.cognitive_architecture.astrocyte_network import AstrocyteNetwork
from .services.chat_service import ChatService
from .adapters.snn_langchain_adapter import SNNLangChainAdapter

# --- ✨新規インポート (生物学的学習則関連) ---
from snn_research.learning_rules import get_bio_learning_rule
from snn_research.bio_models.simple_network import BioSNN
from snn_research.training.bio_trainer import BioTrainer
# ---



def get_auto_device() -> str:
    """実行環境に最適なデバイスを自動的に選択する。"""
    if torch.cuda.is_available(): return "cuda"
    if torch.backends.mps.is_available(): return "mps"
    return "cpu"

def _calculate_t_max(epochs: int, warmup_epochs: int) -> int:
    """学習率スケジューラのT_maxを計算する"""
    return max(1, epochs - warmup_epochs)

def _create_scheduler(optimizer: Optimizer, epochs: int, warmup_epochs: int) -> LRScheduler:
    """ウォームアップ付きのCosineAnnealingスケジューラを生成するファクトリ関数。"""
    warmup_scheduler = LinearLR(optimizer=optimizer, start_factor=1e-3, total_iters=warmup_epochs)
    main_scheduler_t_max = _calculate_t_max(epochs=epochs, warmup_epochs=warmup_epochs)
    main_scheduler = CosineAnnealingLR(optimizer=optimizer, T_max=main_scheduler_t_max)
    return SequentialLR(optimizer=optimizer, schedulers=[warmup_scheduler, main_scheduler], milestones=[warmup_epochs])


class TrainingContainer(containers.DeclarativeContainer):
    """学習に関連するオブジェクトの依存関係を管理するコンテナ。"""
    config = providers.Configuration()

    # --- 共通コンポーネント ---
    tokenizer = providers.Factory(AutoTokenizer.from_pretrained, pretrained_model_name_or_path=config.data.tokenizer_name)
    snn_model = providers.Factory(
        BreakthroughSNN, vocab_size=tokenizer.provided.vocab_size, d_model=config.model.d_model,
        d_state=config.model.d_state, num_layers=config.model.num_layers, time_steps=config.model.time_steps,
        n_head=config.model.n_head, neuron_config=config.model.neuron,
    )
    astrocyte_network = providers.Factory(AstrocyteNetwork, snn_model=snn_model)

    # === 勾配ベース学習 (gradient_based) のためのプロバイダ ===
    optimizer = providers.Factory(AdamW, lr=config.training.gradient_based.learning_rate)
    scheduler = providers.Factory(_create_scheduler, optimizer=optimizer, epochs=config.training.epochs, warmup_epochs=config.training.gradient_based.warmup_epochs)
    standard_loss = providers.Factory(CombinedLoss, tokenizer=tokenizer, **config.training.gradient_based.loss.to_dict())
    distillation_loss = providers.Factory(DistillationLoss, tokenizer=tokenizer, **config.training.gradient_based.distillation.loss.to_dict())
    teacher_model = providers.Factory(AutoModelForCausalLM.from_pretrained, pretrained_model_name_or_path=config.training.gradient_based.distillation.teacher_model)
    standard_trainer = providers.Factory(
        BreakthroughTrainer, model=snn_model, optimizer=optimizer, criterion=standard_loss, scheduler=scheduler,
        device=providers.Factory(get_auto_device), grad_clip_norm=config.training.gradient_based.grad_clip_norm,
        rank=-1, use_amp=config.training.gradient_based.use_amp, log_dir=config.training.log_dir, astrocyte_network=astrocyte_network,
    )
    distillation_trainer = providers.Factory(
        DistillationTrainer, model=snn_model, optimizer=optimizer, criterion=distillation_loss, scheduler=scheduler,
        device=providers.Factory(get_auto_device), grad_clip_norm=config.training.gradient_based.grad_clip_norm,
        rank=-1, use_amp=config.training.gradient_based.use_amp, log_dir=config.training.log_dir, astrocyte_network=astrocyte_network,
    )

    # === 自己教師あり学習 (self_supervised) のためのプロバイダ ===
    ssl_optimizer = providers.Factory(AdamW, lr=config.training.self_supervised.learning_rate)
    ssl_scheduler = providers.Factory(_create_scheduler, optimizer=ssl_optimizer, epochs=config.training.epochs, warmup_epochs=config.training.self_supervised.warmup_epochs)
    self_supervised_loss = providers.Factory(SelfSupervisedLoss, tokenizer=tokenizer, **config.training.self_supervised.loss.to_dict())
    self_supervised_trainer = providers.Factory(
        SelfSupervisedTrainer, model=snn_model, optimizer=ssl_optimizer, criterion=self_supervised_loss, scheduler=ssl_scheduler,
        device=providers.Factory(get_auto_device), grad_clip_norm=config.training.self_supervised.grad_clip_norm,
        rank=-1, use_amp=config.training.self_supervised.use_amp, log_dir=config.training.log_dir, astrocyte_network=astrocyte_network,
    )

    # === 物理情報学習 (physics_informed) のためのプロバイダ ===
    pi_optimizer = providers.Factory(AdamW, lr=config.training.physics_informed.learning_rate)
    pi_scheduler = providers.Factory(_create_scheduler, optimizer=pi_optimizer, epochs=config.training.epochs, warmup_epochs=config.training.physics_informed.warmup_epochs)
    physics_informed_loss = providers.Factory(PhysicsInformedLoss, tokenizer=tokenizer, **config.training.physics_informed.loss.to_dict())
    physics_informed_trainer = providers.Factory(
        PhysicsInformedTrainer, model=snn_model, optimizer=pi_optimizer, criterion=physics_informed_loss, scheduler=pi_scheduler,
        device=providers.Factory(get_auto_device), grad_clip_norm=config.training.physics_informed.grad_clip_norm,
        rank=-1, use_amp=config.training.physics_informed.use_amp, log_dir=config.training.log_dir, astrocyte_network=astrocyte_network,
    )

    # === 生物学的学習 (biologically_plausible) のためのプロバイダ ===
    bio_learning_rule = providers.Factory(get_bio_learning_rule, name=config.training.biologically_plausible.learning_rule, params=config.training.biologically_plausible.to_dict())
    bio_snn_model = providers.Factory(
        BioSNN, n_input=100, n_hidden=50, n_output=10,
        neuron_params=config.training.biologically_plausible.neuron, learning_rule=bio_learning_rule,
    )
    bio_trainer = providers.Factory(BioTrainer, model=bio_snn_model, device=providers.Factory(get_auto_device))

    
    astrocyte_network = providers.Factory(
        AstrocyteNetwork,
        snn_model=snn_model
    )
    
    optimizer = providers.Factory(
        AdamW,
        lr=config.training.gradient_based.learning_rate,
    )
    
    scheduler = providers.Factory(
        _create_scheduler,
        epochs=config.training.epochs,
        warmup_epochs=config.training.gradient_based.warmup_epochs,
    )
    
    standard_loss = providers.Factory(
        CombinedLoss,
        ce_weight=config.training.gradient_based.loss.ce_weight,
        spike_reg_weight=config.training.gradient_based.loss.spike_reg_weight,
        sparsity_reg_weight=config.training.gradient_based.loss.sparsity_reg_weight,
        mem_reg_weight=config.training.gradient_based.loss.mem_reg_weight,
        tokenizer=tokenizer,
    )

    distillation_loss = providers.Factory(
        DistillationLoss,
        ce_weight=config.training.gradient_based.distillation.loss.ce_weight,
        distill_weight=config.training.gradient_based.distillation.loss.distill_weight,
        spike_reg_weight=config.training.gradient_based.distillation.loss.spike_reg_weight,
        sparsity_reg_weight=config.training.gradient_based.distillation.loss.sparsity_reg_weight,
        mem_reg_weight=config.training.gradient_based.distillation.loss.mem_reg_weight,
        temperature=config.training.gradient_based.distillation.loss.temperature,
        tokenizer=tokenizer,
    )
    
    teacher_model = providers.Factory(
        AutoModelForCausalLM.from_pretrained,
        pretrained_model_name_or_path=config.training.gradient_based.distillation.teacher_model
    )

    standard_trainer = providers.Factory(
        BreakthroughTrainer,
        model=snn_model,
        optimizer=optimizer,
        criterion=standard_loss,
        scheduler=scheduler,
        device=providers.Factory(get_auto_device),
        grad_clip_norm=config.training.gradient_based.grad_clip_norm,
        rank=-1, # DDP未使用時のデフォルト値
        use_amp=config.training.gradient_based.use_amp,
        log_dir=config.training.log_dir,
        astrocyte_network=astrocyte_network,
    )

    distillation_trainer = providers.Factory(
        DistillationTrainer,
        model=snn_model,
        optimizer=optimizer,
        criterion=distillation_loss,
        scheduler=scheduler,
        device=providers.Factory(get_auto_device),
        grad_clip_norm=config.training.gradient_based.grad_clip_norm,
        rank=-1,
        use_amp=config.training.gradient_based.use_amp,
        log_dir=config.training.log_dir,
        astrocyte_network=astrocyte_network,
    )

    # === ✨生物学的学習 (biologically_plausible) のためのプロバイダ (新規追加) ===
    bio_learning_rule = providers.Factory(
        get_bio_learning_rule,
        name=config.training.biologically_plausible.learning_rule,
        params={
            "stdp": config.training.biologically_plausible.stdp,
            "reward_modulated_stdp": config.training.biologically_plausible.reward_modulated_stdp,
        }
    )

    bio_snn_model = providers.Factory(
        BioSNN,
        n_input=100,  # ダミーの値, 本来はデータに依存
        n_hidden=50,
        n_output=10,
        neuron_params=config.training.biologically_plausible.neuron,
        learning_rule=bio_learning_rule,
    )
    
    bio_trainer = providers.Factory(
        BioTrainer,
        model=bio_snn_model,
        device=providers.Factory(get_auto_device),
    )

    # === ✨自己教師あり学習 (self_supervised) のためのプロバイダ (新規追加) ===
    ssl_optimizer = providers.Factory(
        AdamW,
        lr=config.training.self_supervised.learning_rate,
    )

    ssl_scheduler = providers.Factory(
        _create_scheduler,
        epochs=config.training.epochs,
        warmup_epochs=config.training.self_supervised.warmup_epochs,
    )
    
    self_supervised_loss = providers.Factory(
        SelfSupervisedLoss,
        prediction_weight=config.training.self_supervised.loss.prediction_weight,
        spike_reg_weight=config.training.self_supervised.loss.spike_reg_weight,
        sparsity_reg_weight=config.training.self_supervised.loss.sparsity_reg_weight,
        mem_reg_weight=config.training.self_supervised.loss.mem_reg_weight,
        tokenizer=tokenizer,
    )

    self_supervised_trainer = providers.Factory(
        SelfSupervisedTrainer,
        model=snn_model,
        optimizer=ssl_optimizer, # SSL専用オプティマイザ
        criterion=self_supervised_loss, # SSL専用損失関数
        scheduler=ssl_scheduler, # SSL専用スケジューラ
        device=providers.Factory(get_auto_device),
        grad_clip_norm=config.training.self_supervised.grad_clip_norm,
        rank=-1,
        use_amp=config.training.self_supervised.use_amp,
        log_dir=config.training.log_dir,
        astrocyte_network=astrocyte_network, # アストロサイトは共用可能
    )


class AppContainer(containers.DeclarativeContainer):
    """GradioアプリやAPIなど、アプリケーション層の依存関係を管理するコンテナ。"""
    config = providers.Configuration()
    device = providers.Factory(lambda cfg_device: get_auto_device() if cfg_device == "auto" else cfg_device, cfg_device=config.inference.device)
    snn_inference_engine = providers.Singleton(SNNInferenceEngine, model_path=config.model.path, device=device)
    chat_service = providers.Factory(ChatService, snn_engine=snn_inference_engine, max_len=config.inference.max_len)
    langchain_adapter = providers.Factory(SNNLangChainAdapter, snn_engine=snn_inference_engine)

