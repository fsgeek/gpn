"""
Bayesian Epistemic Metric (Approach B)

Philosophy: Standard Bayesian uncertainty via MC dropout

Metrics:
- Alignment (same as Simple 2D)
- Correctness (same as Simple 2D)
- Epistemic uncertainty: Variance across MC samples
- Predictive entropy: Entropy of mean predictions

Key Approach:
- Run Witness forward pass N times with dropout enabled
- Compute variance of predictions (epistemic uncertainty)
- High variance = model uncertain about prediction

State Space:
- High epistemic uncertainty + High correctness → Honest "I don't know"
- Low uncertainty + Low correctness → Confident but wrong
- High alignment + High uncertainty → Both unsure (could be collusion or learning)

Pros:
- Principled Bayesian approach
- Established literature
- Directly measures model uncertainty

Cons:
- Computational overhead (~5-10ms with 10 samples)
- Doesn't explicitly detect collusion
- Requires dropout layers in Witness

Implementation Notes:
- Keeps model in .eval() mode to freeze BatchNorm stats
- Manually enables dropout during forward passes
- Uses consistent random seed for reproducibility
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.metrics.epistemic.base import EpistemicMetric, EpistemicState


class BayesianUncertaintyMetric(EpistemicMetric):
    """
    Bayesian metric: MC Dropout uncertainty estimation

    Tracks epistemic state using Bayesian uncertainty quantification.
    """

    def __init__(
        self,
        random_sampling: bool = False,
        sample_rate: float = 0.5,
        device: torch.device | None = None,
        num_mc_samples: int = 10,
    ):
        """
        Initialize Bayesian uncertainty metric.

        Args:
            random_sampling: Enable sparse tensor sampling
            sample_rate: Fraction of metrics to compute when sampling
            device: Device for tensor operations
            num_mc_samples: Number of MC dropout samples for uncertainty
        """
        super().__init__(
            name='bayesian',
            random_sampling=random_sampling,
            sample_rate=sample_rate,
            device=device,
        )

        self.num_mc_samples = num_mc_samples

    def compute(
        self,
        step: int,
        v_pred: torch.Tensor,
        v_seen: torch.Tensor,
        judge_logits: torch.Tensor,
        witness_logits: torch.Tensor,
        labels: torch.Tensor,
        fake_images: torch.Tensor,
        witness_model: nn.Module | None = None,
        **kwargs,
    ) -> EpistemicState:
        """
        Compute Bayesian metrics via MC dropout.

        Args:
            step: Current training step
            v_pred: Weaver's value predictions [B]
            v_seen: Witness's value predictions [B]
            judge_logits: Judge classifier outputs [B, num_classes]
            witness_logits: Witness classifier outputs [B, num_classes]
            labels: Ground truth labels [B]
            fake_images: Generated images [B, C, H, W]
            witness_model: Witness model (for MC dropout)

        Returns:
            EpistemicState with Bayesian uncertainty metrics
        """
        # Base metrics (same as Simple 2D)
        alignment = self.compute_alignment(v_pred, v_seen)
        correctness = self.compute_correctness(judge_logits, labels)
        disagreement = self.compute_disagreement(judge_logits, witness_logits)

        # MC dropout uncertainty (requires witness_model)
        if witness_model is not None and self._should_compute_metric('epistemic_uncertainty'):
            witness_uncertainty, witness_entropy = self._compute_mc_dropout_uncertainty(
                witness_model, fake_images
            )
        else:
            witness_uncertainty = 0.0
            witness_entropy = 0.0

        # Judge uncertainty (no MC dropout - just use logit entropy as proxy)
        judge_entropy = self._compute_entropy(judge_logits)

        # Improvement tracking
        alignment_improvement = self.get_recent_improvement('alignment', window=100)
        correctness_improvement = self.get_recent_improvement('correctness', window=100)

        # State classification
        state = self._classify_state(
            alignment, correctness,
            witness_uncertainty, witness_entropy
        )

        metrics = {
            # Primary dimensions
            'alignment': alignment,
            'correctness': correctness,

            # Bayesian uncertainty
            'witness_epistemic_uncertainty': witness_uncertainty,
            'witness_predictive_entropy': witness_entropy,
            'judge_entropy': judge_entropy,

            # Secondary signals
            'disagreement': disagreement,

            # Improvement tracking
            'alignment_improvement': alignment_improvement,
            'correctness_improvement': correctness_improvement,

            # Derived
            'state': state,
        }

        metadata = {
            'state_name': self._state_name(state),
            'interpretation': self._interpret_state(
                alignment, correctness,
                witness_uncertainty, witness_entropy
            ),
            'num_mc_samples': self.num_mc_samples,
        }

        epistemic_state = EpistemicState(
            step=step,
            approach=self.name,
            metrics=metrics,
            metadata=metadata,
        )

        self.update_history(epistemic_state)
        return epistemic_state

    def _compute_mc_dropout_uncertainty(
        self,
        model: nn.Module,
        images: torch.Tensor,
    ) -> tuple[float, float]:
        """
        Compute epistemic uncertainty via MC dropout.

        Runs model multiple times with dropout enabled to estimate
        predictive uncertainty.

        Args:
            model: Witness model with dropout layers
            images: Input images [B, C, H, W]

        Returns:
            (epistemic_uncertainty, predictive_entropy)
        """
        # Save original training mode
        was_training = model.training

        # Set model to eval mode (freezes BatchNorm stats)
        # but enable dropout manually
        model.eval()

        # Enable dropout for uncertainty estimation
        self._enable_dropout(model)

        # Collect MC samples
        mc_logits = []

        with torch.no_grad():
            for _ in range(self.num_mc_samples):
                # Forward pass with dropout enabled
                logits, _ = model(images)
                mc_logits.append(logits)

        # Stack samples [num_samples, B, num_classes]
        mc_logits = torch.stack(mc_logits, dim=0)

        # Compute epistemic uncertainty (variance of predictions)
        # Variance across MC samples
        probs = F.softmax(mc_logits, dim=2)  # [num_samples, B, num_classes]
        mean_probs = probs.mean(dim=0)  # [B, num_classes]

        # Epistemic uncertainty = variance of predictions
        epistemic_unc = probs.var(dim=0).sum(dim=1).mean().item()

        # Predictive entropy = entropy of mean predictions
        pred_entropy = -(mean_probs * torch.log(mean_probs + 1e-8)).sum(dim=1).mean().item()

        # Restore original training mode
        model.train(was_training)

        return epistemic_unc, pred_entropy

    def _enable_dropout(self, model: nn.Module):
        """
        Enable dropout layers for MC sampling.

        Args:
            model: Model to enable dropout in
        """
        for module in model.modules():
            if isinstance(module, (nn.Dropout, nn.Dropout2d)):
                module.train()

    def _compute_entropy(self, logits: torch.Tensor) -> float:
        """
        Compute entropy of predictions.

        Args:
            logits: Classifier outputs [B, num_classes]

        Returns:
            Average entropy [0, log(num_classes)]
        """
        if not self._should_compute_metric('entropy'):
            return 0.0

        probs = F.softmax(logits, dim=1)
        entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=1).mean().item()
        return entropy

    def _classify_state(
        self,
        alignment: float,
        correctness: float,
        witness_uncertainty: float,
        witness_entropy: float,
        threshold: float = 0.5,
        unc_threshold: float = 0.1,
    ) -> int:
        """
        Classify epistemic state based on uncertainty.

        Args:
            alignment: Alignment score [0, 1]
            correctness: Correctness score [0, 1]
            witness_uncertainty: Epistemic uncertainty
            witness_entropy: Predictive entropy
            threshold: Boundary for high/low correctness
            unc_threshold: Boundary for high/low uncertainty

        Returns:
            State code (0-5):
                0: Low correctness, high uncertainty → Honest "I don't know"
                1: Low correctness, low uncertainty → Confident but wrong
                2: High correctness, high uncertainty → Correct but unsure
                3: High correctness, low uncertainty → Confident and correct
                4: High alignment, low correctness → Collusion risk
                5: Other/transitional
        """
        high_correct = correctness > threshold
        high_unc = witness_uncertainty > unc_threshold
        high_align = alignment > threshold

        if not high_correct and high_unc:
            return 0  # Honest "I don't know"
        elif not high_correct and not high_unc:
            if high_align:
                return 4  # Collusion (both wrong, both confident, aligned)
            else:
                return 1  # Confident but wrong
        elif high_correct and high_unc:
            return 2  # Correct but unsure (learning)
        elif high_correct and not high_unc:
            return 3  # Confident and correct (good)
        else:
            return 5  # Other/transitional

    def _state_name(self, state: int) -> str:
        """Get human-readable state name."""
        names = {
            0: 'honest_uncertainty',
            1: 'confident_wrong',
            2: 'correct_unsure',
            3: 'confident_correct',
            4: 'collusion_risk',
            5: 'transitional',
        }
        return names.get(state, 'unknown')

    def _interpret_state(
        self,
        alignment: float,
        correctness: float,
        witness_uncertainty: float,
        witness_entropy: float,
    ) -> str:
        """
        Generate human-readable interpretation.

        Args:
            alignment: Alignment score
            correctness: Correctness score
            witness_uncertainty: Epistemic uncertainty
            witness_entropy: Predictive entropy

        Returns:
            Interpretation string
        """
        state = self._classify_state(
            alignment, correctness,
            witness_uncertainty, witness_entropy
        )

        if state == 0:
            return "Honest uncertainty: High epistemic uncertainty, appropriately uncertain"
        elif state == 1:
            return "WARNING: Confident but wrong - low uncertainty despite low correctness"
        elif state == 2:
            return "Learning: Correct but still uncertain (healthy)"
        elif state == 3:
            return "Good: Confident and correct"
        elif state == 4:
            return "WARNING: Collusion risk - aligned, confident, but wrong"
        else:
            return "Transitional state"
