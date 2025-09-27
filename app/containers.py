# matsushibadenki/snn/app/containers.py
# DIコンテナの定義ファイル
# 
# 機能:
# - プロジェクト全体の依存関係を一元管理する。
# - 設定ファイルに基づいてオブジェクトを生成・設定する。
# - 学習用とアプリ用のコンテナを分離し、関心を分離。
# - 独自Vocabularyを廃止し、Hugging Face Tokenizerに全面的に移行。
# - トークナイザの読み込み元をdistillation設定から共通設定に変更。
# - 損失関数にpad_idではなくtokenizerプロバイダを渡すように修正し、依存関係の解決を遅延させる。
# - スケジューラの依存関係問題を解決するため、メソッドから独立した関数ファクトリにリファクタリング。
# - Trainerの定義にuse_ampとlog_dirを追加。

from dependency_injector import containers, providers
from torch.optim import AdamW, Optimizer
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR, LRScheduler
from transformers import AutoTokenizer, AutoModelForCausalLM

# プロジェクト内モジュールのインポート
from snn_research.core.snn_core import BreakthroughSNN
from snn_research.deployment import SNNInferenceEngine
from snn_research.training.losses import CombinedLoss, DistillationLoss
from snn_research.training.trainers import BreakthroughTrainer, DistillationTrainer
from .services.chat_service import ChatService
from .adapters.snn_langchain_adapter import SNNLangChainAdapter

def _calculate_t_max(epochs: int, warmup_epochs: int) -> int:
    """学習率スケジューラのT_maxを計算する"""
    return max(1, epochs - warmup_epochs)

def _create_scheduler(optimizer: Optimizer, epochs: int, warmup_epochs: int) -> LRScheduler:
    """ウォームアップ付きのCosineAnnealingスケジューラを生成するファクトリ関数。"""
    warmup_scheduler = LinearLR(
        optimizer=optimizer,
        start_factor=1e-3,
        total_iters=warmup_epochs,
    )
    
    main_scheduler_t_max = _calculate_t_max(
        epochs=epochs,
        warmup_epochs=warmup_epochs,
    )

    main_scheduler = CosineAnnealingLR(
        optimizer=optimizer,
        T_max=main_scheduler_t_max,
    )

    return SequentialLR(
        optimizer=optimizer,
        schedulers=[warmup_scheduler, main_scheduler],
        milestones=[warmup_epochs],
    )

class TrainingContainer(containers.DeclarativeContainer):
    """学習に関連するオブジェクトの依存関係を管理するコンテナ。"""
    config = providers.Configuration()

    # --- トークナイザ ---
    tokenizer = providers.Factory(
        AutoTokenizer.from_pretrained,
        pretrained_model_name_or_path=config.data.tokenizer_name
    )

    # --- モデル関連 ---
    snn_model = providers.Factory(
        BreakthroughSNN,
        vocab_size=tokenizer.provided.vocab_size,
        d_model=config.model.d_model,
        d_state=config.model.d_state,
        num_layers=config.model.num_layers,
        time_steps=config.model.time_steps,
        n_head=config.model.n_head,
    )

    # --- 学習コンポーネント ---
    optimizer = providers.Factory(
        AdamW,
        lr=config.training.learning_rate,
    )
    
    # --- 学習率スケジューラ ---
    # 独立したファクトリ関数をプロバイダとして登録
    scheduler = providers.Factory(
        _create_scheduler,
        epochs=config.training.epochs,
        warmup_epochs=config.training.warmup_epochs,
    )
    
    # --- 損失関数 ---
    standard_loss = providers.Factory(
        CombinedLoss,
        ce_weight=config.training.loss.ce_weight,
        spike_reg_weight=config.training.loss.spike_reg_weight,
        tokenizer=tokenizer,
    )
    distillation_loss = providers.Factory(
        DistillationLoss,
        ce_weight=config.training.distillation.loss.ce_weight,
        distill_weight=config.training.distillation.loss.distill_weight,
        spike_reg_weight=config.training.distillation.loss.spike_reg_weight,
        temperature=config.training.distillation.loss.temperature,
        tokenizer=tokenizer,
    )
    
    # --- 蒸留用教師モデル ---
    teacher_model = providers.Factory(
        AutoModelForCausalLM.from_pretrained,
        pretrained_model_name_or_path=config.training.distillation.teacher_model
    )

    # --- トレーナー定義 ---
    standard_trainer = providers.Factory(
        BreakthroughTrainer,
        criterion=standard_loss,
        grad_clip_norm=config.training.grad_clip_norm,
        use_amp=config.training.use_amp,
        log_dir=config.training.log_dir,
    )

    distillation_trainer = providers.Factory(
        DistillationTrainer,
        criterion=distillation_loss,
        grad_clip_norm=config.training.grad_clip_norm,
        use_amp=config.training.use_amp,
        log_dir=config.training.log_dir,
    )


class AppContainer(containers.DeclarativeContainer):
    """GradioアプリやAPIなど、アプリケーション層の依存関係を管理するコンテナ。"""
    config = providers.Configuration()

    # --- 推論エンジン ---
    snn_inference_engine = providers.Singleton(
        SNNInferenceEngine,
        model_path=config.model.path,
        device=config.inference.device,
    )
    
    # --- サービス ---
    chat_service = providers.Factory(
        ChatService,
        snn_engine=snn_inference_engine,
        max_len=config.inference.max_len,
    )
    
    # --- LangChainアダプタ ---
    langchain_adapter = providers.Factory(
        SNNLangChainAdapter,
        snn_engine=snn_inference_engine,
    )
