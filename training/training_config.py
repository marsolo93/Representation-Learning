from dataclasses import dataclass
from typing import Literal
from models import ViTConfig
from scheduler import Scheduler


@dataclass
class DINOV1Config:
    num_locals: int
    vit_config: ViTConfig
    teacher_temp: float
    student_temp: float
    learning_rate: float
    weight_decay: float
    gradient_clipping: float  # TODO: check the gradient clipping
    learning_rate_scheduler: Scheduler | None = None
    weight_decay_scheduler: Scheduler | None = None
    teacher_temp_scheduler: Scheduler | None = None
    teacher_centering_option: Literal['update', 'sinkhorn_kappa'] = 'update'


@dataclass
class DINOV2Config:
    num_locals: int
    vit_config: ViTConfig
    teacher_temp: float
    student_temp: float
    learning_rate: float
    weight_decay: float
    gradient_clipping: float  # TODO: check the gradient clipping
    dino_loss_weight: float
    ibot_loss_weight: float
    learning_rate_scheduler: Scheduler | None = None
    weight_decay_scheduler: Scheduler | None = None
    teacher_temp_scheduler: Scheduler | None = None
    teacher_centering_option: Literal['update', 'sinkhorn_kappa'] = 'update'


# TODO: prepare the DINOV3Config
@dataclass
class DINOV3Config:
    num_locals: int
    vit_config: ViTConfig
    teacher_temp: float
    student_temp: float
    learning_rate: float
    weight_decay: float
    gradient_clipping: float  # TODO: check the gradient clipping
    dino_loss_weight: float
    ibot_loss_weight: float
    gram_loss_weight: float
    learning_rate_scheduler: Scheduler | None = None
    weight_decay_scheduler: Scheduler | None = None
    teacher_temp_scheduler: Scheduler | None = None
    teacher_centering_option: Literal['update', 'sinkhorn_kappa'] = 'update'
