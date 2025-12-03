"""
Loss functions for GPN-1 training.

Implements the three core losses per data-model.md:
    - grounding_loss: Cross-entropy with frozen Judge
    - alignment_loss: MSE between v_pred and v_seen
    - empowerment_loss: Goldilocks KL with variance shrinkage penalty

Exports:
    - grounding_loss
    - alignment_loss
    - empowerment_loss
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def grounding_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    judge_logits: torch.Tensor,
) -> torch.Tensor:
    """
    Compute grounding loss using frozen Judge as reference.

    The grounding loss ensures generated images are classified correctly
    by the external Judge. This keeps the Weaver/Witness system grounded
    in actual digit recognition.

    Per data-model.md: Cross-entropy between Witness logits and Judge's predictions.

    Args:
        logits: Witness classification logits [B, 10]
        labels: Target labels [B]
        judge_logits: Judge classification logits [B, 10] (from frozen Judge)

    Returns:
        Scalar grounding loss

    Note:
        The Judge must be frozen during training. We use its predictions
        as soft targets via KL divergence for smoother gradients.
    """
    # Convert Judge logits to probabilities (soft targets)
    judge_probs = F.softmax(judge_logits.detach(), dim=1)

    # Witness log probabilities
    log_probs = F.log_softmax(logits, dim=1)

    # KL divergence: Judge tells Witness what to aim for
    kl_loss = F.kl_div(log_probs, judge_probs, reduction='batchmean')

    # Also add hard label cross-entropy for direct grounding
    ce_loss = F.cross_entropy(logits, labels)

    # Combine (weighted towards soft targets for smoother training)
    return 0.5 * kl_loss + 0.5 * ce_loss


def alignment_loss(
    v_pred: torch.Tensor,
    v_seen: torch.Tensor,
) -> torch.Tensor:
    """
    Compute alignment loss between Weaver's prediction and Witness's perception.

    The alignment loss trains the Weaver to predict what value the Witness
    will perceive in its generated images. This creates a cooperative signal
    where the Weaver learns to understand the Witness's perspective.

    Per data-model.md: MSE between v_pred and v_seen.

    Args:
        v_pred: Weaver's value prediction [B, v_dim]
        v_seen: Witness's value estimation [B, v_dim] (detached for Weaver training)

    Returns:
        Scalar alignment loss

    Note:
        When training the Weaver, v_seen should be detached to prevent
        gradients flowing through the Witness. When training jointly,
        both can receive gradients.
    """
    return F.mse_loss(v_pred, v_seen)


def empowerment_loss(
    v_pred: torch.Tensor,
    v_seen: torch.Tensor,
    ema_mean: torch.Tensor,
    ema_var: torch.Tensor,
    target_kl: float = 0.5,
    tolerance: float = 0.1,
) -> torch.Tensor:
    """
    Compute Goldilocks empowerment loss.

    The empowerment loss encourages the Weaver to produce images that are
    neither too predictable (boring) nor too unpredictable (chaotic).
    It uses EMA statistics to track the "normal" range and penalizes
    deviations that are too small (variance collapse) or too large.

    Per data-model.md: Goldilocks KL with variance shrinkage penalty.

    Args:
        v_pred: Weaver's value prediction [B, v_dim]
        v_seen: Witness's value estimation [B, v_dim]
        ema_mean: EMA of v_seen mean [v_dim]
        ema_var: EMA of v_seen variance [v_dim]
        target_kl: Target KL divergence (default 0.5)
        tolerance: Acceptable range around target (default 0.1)

    Returns:
        Scalar empowerment loss

    Note:
        This loss should only be active in Phase 2. In Phase 1 and Phase 3,
        return zero to disable empowerment signaling.
    """
    # Compute current batch statistics
    batch_mean = v_seen.mean(dim=0)
    batch_var = v_seen.var(dim=0, unbiased=False) + 1e-8

    # Compute KL divergence between current batch and EMA distribution
    # Assuming Gaussian distributions
    # KL(p||q) = 0.5 * (var_p/var_q + (mean_q - mean_p)^2/var_q - 1 + log(var_q/var_p))
    kl_div = 0.5 * (
        batch_var / (ema_var + 1e-8)
        + (ema_mean - batch_mean).pow(2) / (ema_var + 1e-8)
        - 1
        + torch.log((ema_var + 1e-8) / batch_var)
    ).mean()

    # Goldilocks penalty: penalize being too close or too far from target
    lower_bound = target_kl - tolerance
    upper_bound = target_kl + tolerance

    if kl_div < lower_bound:
        # Too predictable - encourage more diversity
        penalty = (lower_bound - kl_div).pow(2)
    elif kl_div > upper_bound:
        # Too chaotic - encourage more consistency
        penalty = (kl_div - upper_bound).pow(2)
    else:
        # In the sweet spot
        penalty = torch.tensor(0.0, device=v_pred.device)

    # Variance shrinkage penalty
    # Penalize if variance is collapsing (too small compared to EMA)
    var_ratio = batch_var / (ema_var + 1e-8)
    shrinkage_penalty = F.relu(0.1 - var_ratio).mean()  # Penalize if < 10% of EMA var

    return penalty + 0.5 * shrinkage_penalty


class CombinedLoss(nn.Module):
    """
    Combined loss for GPN training with phase-aware weighting.

    Combines grounding, alignment, and empowerment losses with
    weights that vary by training phase.
    """

    def __init__(
        self,
        grounding_weight: float = 1.0,
        alignment_weight: float = 0.1,
        empowerment_weight: float = 0.0,
        target_kl: float = 0.5,
        tolerance: float = 0.1,
    ) -> None:
        """
        Initialize combined loss.

        Args:
            grounding_weight: Weight for grounding loss
            alignment_weight: Weight for alignment loss
            empowerment_weight: Weight for empowerment loss
            target_kl: Target KL for empowerment
            tolerance: Tolerance for empowerment
        """
        super().__init__()
        self.grounding_weight = grounding_weight
        self.alignment_weight = alignment_weight
        self.empowerment_weight = empowerment_weight
        self.target_kl = target_kl
        self.tolerance = tolerance

    def forward(
        self,
        witness_logits: torch.Tensor,
        labels: torch.Tensor,
        judge_logits: torch.Tensor,
        v_pred: torch.Tensor,
        v_seen: torch.Tensor,
        ema_mean: torch.Tensor | None = None,
        ema_var: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """
        Compute combined loss.

        Args:
            witness_logits: Witness classification logits [B, 10]
            labels: Target labels [B]
            judge_logits: Judge classification logits [B, 10]
            v_pred: Weaver's value prediction [B, v_dim]
            v_seen: Witness's value estimation [B, v_dim]
            ema_mean: EMA mean for empowerment (optional)
            ema_var: EMA variance for empowerment (optional)

        Returns:
            Tuple of (total_loss, loss_components_dict)
        """
        components = {}

        # Grounding loss
        if self.grounding_weight > 0:
            g_loss = grounding_loss(witness_logits, labels, judge_logits)
            components["grounding"] = g_loss
        else:
            g_loss = torch.tensor(0.0, device=v_pred.device)
            components["grounding"] = g_loss

        # Alignment loss
        if self.alignment_weight > 0:
            a_loss = alignment_loss(v_pred, v_seen.detach())
            components["alignment"] = a_loss
        else:
            a_loss = torch.tensor(0.0, device=v_pred.device)
            components["alignment"] = a_loss

        # Empowerment loss
        if self.empowerment_weight > 0 and ema_mean is not None and ema_var is not None:
            e_loss = empowerment_loss(
                v_pred, v_seen, ema_mean, ema_var,
                self.target_kl, self.tolerance
            )
            components["empowerment"] = e_loss
        else:
            e_loss = torch.tensor(0.0, device=v_pred.device)
            components["empowerment"] = e_loss

        # Weighted sum
        total = (
            self.grounding_weight * g_loss
            + self.alignment_weight * a_loss
            + self.empowerment_weight * e_loss
        )
        components["total"] = total

        return total, components

    def update_weights(
        self,
        grounding: float | None = None,
        alignment: float | None = None,
        empowerment: float | None = None,
    ) -> None:
        """Update loss weights (for phase transitions)."""
        if grounding is not None:
            self.grounding_weight = grounding
        if alignment is not None:
            self.alignment_weight = alignment
        if empowerment is not None:
            self.empowerment_weight = empowerment
