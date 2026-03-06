from typing import Optional, Union, Callable

import numpy as np
import torch
import torch.nn as nn

from cleandiffuser.classifier import BaseClassifier
from cleandiffuser.nn_condition import BaseNNCondition
from cleandiffuser.nn_diffusion import BaseNNDiffusion
from cleandiffuser.utils import at_least_ndim
from .basic import DiffusionModel


class ShortcutModel(DiffusionModel):
    """Shortcut Model for one-step (or few-step) trajectory generation.

    Based on "One Step Diffusion via Shortcut Models" (Frans et al., 2024).
    https://arxiv.org/abs/2410.12557

    Shortcut models extend flow matching by conditioning the network on both the
    current noise level `t` and the desired step size `dt`. This allows the model
    to "skip ahead" during generation — at inference, setting dt=1.0 generates
    a full trajectory in a single forward pass.

    Training combines two objectives:
        1. Flow matching (small dt): standard velocity regression v = x0 - x1
        2. Bootstrap shortcut (larger dt): run model twice at dt/2, use averaged
           velocity as the target for dt. No separate distillation phase needed.

    The `dt` signal is encoded and injected into the existing `condition` embedding,
    so the DiT1d backbone requires no architectural changes.

    Args:
        nn_diffusion: BaseNNDiffusion
            The neural network backbone (e.g. DiT1d). Must accept
            (x, t, condition) where condition has dim `emb_dim`.
        nn_condition: Optional[BaseNNCondition]
            Optional condition encoder for classifier-free guidance.
        fix_mask: array-like
            Fix a portion of the input; only the unmasked part is generated.
        loss_weight: array-like
            Per-element loss weighting.
        grad_clip_norm: Optional[float]
            Gradient clipping norm.
        ema_rate: float
            EMA decay rate for the inference model.
        optim_params: Optional[dict]
            AdamW optimizer parameters.
        emb_dim: int
            Embedding dimension for dt. Must match the emb_dim of nn_diffusion.
        num_shortcut_levels: int
            Number of dt levels used during training. dt is sampled as
            2^(-k) for k in [0, num_shortcut_levels). Default: 4 gives
            dt in {1.0, 0.5, 0.25, 0.125}.
        flow_matching_weight: float
            Fraction of the batch trained with the flow matching objective.
            The rest uses the bootstrap shortcut objective. Default: 0.5.
        device: Union[torch.device, str]
            Device to run on.
    """

    def __init__(
            self,

            # ----------------- Neural Networks ----------------- #
            nn_diffusion: BaseNNDiffusion,
            nn_condition: Optional[BaseNNCondition] = None,

            # ----------------- Masks ----------------- #
            fix_mask: Union[list, np.ndarray, torch.Tensor] = None,
            loss_weight: Union[list, np.ndarray, torch.Tensor] = None,

            # ------------------ Training Params ---------------- #
            grad_clip_norm: Optional[float] = None,
            ema_rate: float = 0.995,
            optim_params: Optional[dict] = None,

            # ------------------- Shortcut Params ------------------- #
            emb_dim: int = 128,
            num_shortcut_levels: int = 4,
            flow_matching_weight: float = 0.5,

            x_max: Optional[torch.Tensor] = None,
            x_min: Optional[torch.Tensor] = None,

            device: Union[torch.device, str] = "cpu"
    ):
        super().__init__(
            nn_diffusion, nn_condition, fix_mask, loss_weight, None,
            grad_clip_norm, 0, ema_rate, optim_params, device)

        self.emb_dim = emb_dim
        self.num_shortcut_levels = num_shortcut_levels
        self.flow_matching_weight = flow_matching_weight
        self.x_max, self.x_min = x_max, x_min

        # Small MLP to embed dt into the same space as the time/condition embedding.
        # Output is added to condition before being passed to nn_diffusion,
        # so DiT1d requires no changes.
        self.map_dt = nn.Sequential(
            nn.Linear(emb_dim, emb_dim), nn.Mish(),
            nn.Linear(emb_dim, emb_dim)
        ).to(device)

        # TODO: decide on sinusoidal vs learned embedding for dt.
        # A sinusoidal embedding of dt (scalar in [0,1]) before map_dt is likely best.
        # See cleandiffuser/utils for SinusoidalEmbedding reference.

    @property
    def clip_pred(self):
        return (self.x_max is not None) or (self.x_min is not None)

    def _encode_dt(self, dt: torch.Tensor) -> torch.Tensor:
        """Encode scalar dt values into embedding space.

        Args:
            dt: (b,) tensor of step sizes in [0, 1].
        Returns:
            (b, emb_dim) embedding to be added to the condition vector.
        """
        # TODO: embed dt using sinusoidal encoding, then project through map_dt.
        # Example: use SinusoidalEmbedding(emb_dim)(dt) then self.map_dt(...)
        raise NotImplementedError

    def _get_condition_with_dt(
            self,
            model: nn.ModuleDict,
            condition_cfg,
            mask_cfg,
            dt: torch.Tensor
    ) -> Optional[torch.Tensor]:
        """Encode the task condition and inject dt embedding into it.

        The dt embedding is added to the condition vector so DiT1d sees a single
        fused embedding: condition_vec + dt_emb.

        Args:
            model: the model (or model_ema) ModuleDict.
            condition_cfg: raw condition input (state, goal, etc.) or None.
            mask_cfg: CFG mask or None.
            dt: (b,) step size tensor.
        Returns:
            Fused condition tensor of shape (b, emb_dim), or just dt_emb if
            no task condition is provided.
        """
        # TODO: implement fusion of task condition + dt embedding.
        raise NotImplementedError

    # ==================== Training ====================

    def loss(self, x0, condition=None, x1=None, dt=None):
        """Compute the shortcut training loss.

        Combines:
          - Flow matching objective (fraction `flow_matching_weight` of batch)
          - Bootstrap shortcut objective (remaining fraction)

        Args:
            x0: (b, horizon, dim) target trajectories from the dataset.
            condition: optional task condition.
            x1: optional source noise. Defaults to standard Gaussian.
            dt: optional step size. If None, sampled randomly per the shortcut
                level schedule.
        Returns:
            Scalar loss tensor.
        """
        # TODO: implement combined flow matching + bootstrap shortcut loss.
        # Steps:
        #   1. Split batch into flow-matching half and shortcut half.
        #   2. Flow matching half:
        #      - Sample t ~ Uniform[0, 1], interpolate xt = x0 + t*(x1 - x0)
        #      - dt_fm = 0 (or very small), target velocity = x0 - x1
        #      - Loss = MSE(model(xt, t, condition+dt_emb), target)
        #   3. Bootstrap shortcut half:
        #      - Sample dt level k, so dt = 2^(-k)
        #      - Sample t, compute xt
        #      - Run model twice at dt/2 with no_grad to get bootstrap velocity target
        #      - Loss = MSE(model(xt, t, condition+dt_emb(dt)), bootstrap_target)
        raise NotImplementedError

    def update(self, x0, condition=None, update_ema=True, x1=None, **kwargs):
        """One gradient update step.

        Args:
            x0: (b, horizon, dim) samples from the target distribution.
            condition: optional task condition.
            update_ema: whether to update the EMA model after the step.
            x1: optional source noise samples.
        Returns:
            log dict with 'loss' and 'grad_norm'.
        """
        loss = self.loss(x0, condition, x1)

        loss.backward()
        grad_norm = nn.utils.clip_grad_norm_(
            list(self.model.parameters()) + list(self.map_dt.parameters()),
            self.grad_clip_norm
        ) if self.grad_clip_norm else None
        self.optimizer.step()
        self.optimizer.zero_grad()

        if update_ema:
            self.ema_update()

        return {"loss": loss.item(), "grad_norm": grad_norm}

    # ==================== Sampling ====================

    def sample(
            self,
            # ---------- the known fixed portion ---------- #
            prior: torch.Tensor,
            # ----------------- sampling ----------------- #
            n_samples: int = 1,
            sample_steps: int = 1,
            use_ema: bool = True,
            temperature: float = 1.0,
            # ------------------ guidance ------------------ #
            condition_cfg=None,
            mask_cfg=None,
            w_cfg: float = 0.0,
            # ------------------ others ------------------ #
            requires_grad: bool = False,
            preserve_history: bool = False,
            **kwargs,
    ):
        """Generate samples using shortcut Euler stepping.

        At sample_steps=1, dt=1.0: single forward pass from noise to data.
        At sample_steps=k: dt=1/k, k Euler steps (standard flow matching).

        Args:
            prior: (n_samples, horizon, dim) fixed portion of the input.
            n_samples: number of samples to generate.
            sample_steps: number of Euler steps. 1 = one-step generation.
            use_ema: use EMA model for inference.
            temperature: noise scale.
            condition_cfg: task condition for CFG.
            mask_cfg: CFG mask.
            w_cfg: CFG guidance weight.
            requires_grad: preserve gradients through sampling.
            preserve_history: store intermediate denoising states.
        Returns:
            (x0, log) — generated trajectories and log dict.
        """
        # TODO: implement Euler sampling loop.
        # dt = 1.0 / sample_steps
        # x1 = randn_like(prior) * temperature
        # for each step:
        #   vel = model(xt, t, condition + dt_emb(dt))
        #   xt = xt + dt * vel
        #   apply fix_mask
        # optionally clip predictions
        raise NotImplementedError
