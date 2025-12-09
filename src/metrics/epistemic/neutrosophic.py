"""
Neutrosophic Epistemic Metric (Approach C)

Philosophy: Three-way decomposition per GPN-2 spec

Neutrosophic Logic {T, I, F}:
- T (Truth): Genuine synchronization (alignment + correctness + improvement)
- I (Indeterminacy): Honest uncertainty (weaver_unc + witness_entropy + disagreement)
- F (Falsity): Gaming/collusion (collusion + mode_collapse + gaming)

Metrics:
- T = 0.4×alignment + 0.4×correctness + 0.2×improvement
- I = 0.3×weaver_unc + 0.3×witness_entropy + 0.4×disagreement
- F = 0.4×collusion + 0.3×mode_collapse + 0.3×gaming

State Space:
- High T, Low I, Low F → Healthy learning
- Low T, High I, Low F → Early training (honest uncertainty)
- Low T, Low I, High F → Collusion/gaming

Pros:
- Explicitly separates all three epistemic states
- Matches GPN-2 design philosophy
- Fine-grained diagnostic signals

Cons:
- More complex computation (~0.5-1ms overhead)
- Weight choices somewhat arbitrary
- Never validated (hypothesis to test)
"""

import torch
import torch.nn.functional as F

from src.metrics.epistemic.base import EpistemicMetric, EpistemicState


class NeutrosophicMetric(EpistemicMetric):
    """
    Neutrosophic metric: {T, I, F} decomposition

    Tracks epistemic state using three-valued logic.
    """

    def __init__(
        self,
        random_sampling: bool = False,
        sample_rate: float = 0.5,
        device: torch.device | None = None,
        # Weight configurations
        t_weights: tuple[float, float, float] = (0.4, 0.4, 0.2),
        i_weights: tuple[float, float, float] = (0.3, 0.3, 0.4),
        f_weights: tuple[float, float, float] = (0.4, 0.3, 0.3),
    ):
        """
        Initialize Neutrosophic metric.

        Args:
            random_sampling: Enable sparse tensor sampling
            sample_rate: Fraction of metrics to compute when sampling
            device: Device for tensor operations
            t_weights: Weights for T (alignment, correctness, improvement)
            i_weights: Weights for I (weaver_unc, witness_entropy, disagreement)
            f_weights: Weights for F (collusion, mode_collapse, gaming)
        """
        super().__init__(
            name='neutrosophic',
            random_sampling=random_sampling,
            sample_rate=sample_rate,
            device=device,
        )

        # Store weights (should sum to 1.0 for each)
        self.t_weights = t_weights
        self.i_weights = i_weights
        self.f_weights = f_weights

        # Validate weights
        assert abs(sum(t_weights) - 1.0) < 1e-6, "T weights must sum to 1.0"
        assert abs(sum(i_weights) - 1.0) < 1e-6, "I weights must sum to 1.0"
        assert abs(sum(f_weights) - 1.0) < 1e-6, "F weights must sum to 1.0"

    def compute(
        self,
        step: int,
        v_pred: torch.Tensor,
        v_seen: torch.Tensor,
        judge_logits: torch.Tensor,
        witness_logits: torch.Tensor,
        labels: torch.Tensor,
        fake_images: torch.Tensor,
        **kwargs,
    ) -> EpistemicState:
        """
        Compute Neutrosophic {T, I, F} metrics.

        Args:
            step: Current training step
            v_pred: Weaver's value predictions [B]
            v_seen: Witness's value predictions [B]
            judge_logits: Judge classifier outputs [B, num_classes]
            witness_logits: Witness classifier outputs [B, num_classes]
            labels: Ground truth labels [B]
            fake_images: Generated images [B, C, H, W]

        Returns:
            EpistemicState with T, I, F components
        """
        # Base metrics
        alignment = self.compute_alignment(v_pred, v_seen)
        correctness = self.compute_correctness(judge_logits, labels)
        disagreement = self.compute_disagreement(judge_logits, witness_logits)

        # Truth (T) components
        improvement = self._compute_improvement()
        t_value = self._compute_truth(alignment, correctness, improvement)

        # Indeterminacy (I) components
        weaver_unc = self._compute_weaver_uncertainty(v_pred)
        witness_entropy = self._compute_witness_entropy(witness_logits)
        i_value = self._compute_indeterminacy(weaver_unc, witness_entropy, disagreement)

        # Falsity (F) components
        collusion = self._compute_collusion(alignment, correctness)
        mode_collapse = self._compute_mode_collapse(judge_logits, labels)
        gaming = self._compute_gaming(alignment, correctness)
        f_value = self._compute_falsity(collusion, mode_collapse, gaming)

        # State classification
        state_class = self._classify_state(t_value, i_value, f_value)

        metrics = {
            # Primary neutrosophic values
            'T': t_value,
            'I': i_value,
            'F': f_value,

            # Truth components
            'alignment': alignment,
            'correctness': correctness,
            'improvement': improvement,

            # Indeterminacy components
            'weaver_uncertainty': weaver_unc,
            'witness_entropy': witness_entropy,
            'disagreement': disagreement,

            # Falsity components
            'collusion': collusion,
            'mode_collapse': mode_collapse,
            'gaming': gaming,

            # Derived
            'state_class': state_class,
        }

        metadata = {
            'state_name': self._state_name(state_class),
            'interpretation': self._interpret_state(t_value, i_value, f_value),
            'dominant_component': self._dominant_component(t_value, i_value, f_value),
        }

        state = EpistemicState(
            step=step,
            approach=self.name,
            metrics=metrics,
            metadata=metadata,
        )

        self.update_history(state)
        return state

    # ========================================================================
    # Truth (T) computation
    # ========================================================================

    def _compute_improvement(self) -> float:
        """
        Compute recent improvement in alignment and correctness.

        Returns:
            Normalized improvement [0, 1]
        """
        if not self._should_compute_metric('improvement'):
            return 0.0

        alignment_improvement = self.get_recent_improvement('alignment', window=100)
        correctness_improvement = self.get_recent_improvement('correctness', window=100)

        # Average improvement, normalized to [0, 1]
        avg_improvement = (alignment_improvement + correctness_improvement) / 2.0
        # Clamp to [0, 1] (negative improvement becomes 0)
        improvement = max(0.0, min(1.0, avg_improvement + 0.5))
        return improvement

    def _compute_truth(
        self,
        alignment: float,
        correctness: float,
        improvement: float,
    ) -> float:
        """
        Compute T (Truth) value.

        T = 0.4×alignment + 0.4×correctness + 0.2×improvement

        Args:
            alignment: Alignment score [0, 1]
            correctness: Correctness score [0, 1]
            improvement: Improvement score [0, 1]

        Returns:
            T value [0, 1]
        """
        t_value = (
            self.t_weights[0] * alignment +
            self.t_weights[1] * correctness +
            self.t_weights[2] * improvement
        )
        return t_value

    # ========================================================================
    # Indeterminacy (I) computation
    # ========================================================================

    def _compute_weaver_uncertainty(self, v_pred: torch.Tensor) -> float:
        """
        Compute Weaver's uncertainty from value prediction variance.

        High variance indicates Weaver is uncertain.

        Args:
            v_pred: Weaver's value predictions [B]

        Returns:
            Normalized uncertainty [0, 1]
        """
        if not self._should_compute_metric('weaver_uncertainty'):
            return 0.0

        variance = v_pred.var().item()
        # Normalize: assume reasonable variance range is [0, 0.25]
        uncertainty = min(1.0, variance / 0.25)
        return uncertainty

    def _compute_witness_entropy(self, witness_logits: torch.Tensor) -> float:
        """
        Compute Witness prediction entropy.

        High entropy indicates Witness is uncertain.

        Args:
            witness_logits: Witness classifier outputs [B, num_classes]

        Returns:
            Normalized entropy [0, 1]
        """
        if not self._should_compute_metric('witness_entropy'):
            return 0.0

        probs = F.softmax(witness_logits, dim=1)
        entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=1).mean().item()

        # Normalize by max entropy (log(num_classes))
        num_classes = witness_logits.size(1)
        max_entropy = torch.log(torch.tensor(num_classes)).item()
        normalized_entropy = entropy / max_entropy
        return normalized_entropy

    def _compute_indeterminacy(
        self,
        weaver_unc: float,
        witness_entropy: float,
        disagreement: float,
    ) -> float:
        """
        Compute I (Indeterminacy) value.

        I = 0.3×weaver_unc + 0.3×witness_entropy + 0.4×disagreement

        Args:
            weaver_unc: Weaver uncertainty [0, 1]
            witness_entropy: Witness entropy [0, 1]
            disagreement: Judge-Witness disagreement [0, 1]

        Returns:
            I value [0, 1]
        """
        i_value = (
            self.i_weights[0] * weaver_unc +
            self.i_weights[1] * witness_entropy +
            self.i_weights[2] * disagreement
        )
        return i_value

    # ========================================================================
    # Falsity (F) computation
    # ========================================================================

    def _compute_collusion(self, alignment: float, correctness: float) -> float:
        """
        Compute collusion score.

        Collusion = high alignment AND low correctness (both wrong but agreeing).

        Args:
            alignment: Alignment score [0, 1]
            correctness: Correctness score [0, 1]

        Returns:
            Collusion score [0, 1]
        """
        if not self._should_compute_metric('collusion'):
            return 0.0

        # High alignment, low correctness
        collusion = alignment * (1.0 - correctness)
        return collusion

    def _compute_mode_collapse(
        self,
        judge_logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> float:
        """
        Compute mode collapse score.

        Mode collapse = low diversity in predicted classes.

        Args:
            judge_logits: Judge classifier outputs [B, num_classes]
            labels: Ground truth labels [B] (for reference)

        Returns:
            Mode collapse score [0, 1]
        """
        if not self._should_compute_metric('mode_collapse'):
            return 0.0

        preds = judge_logits.argmax(dim=1)
        num_classes = judge_logits.size(1)

        # Compute distribution over predicted classes
        pred_counts = torch.bincount(preds, minlength=num_classes).float()
        pred_dist = pred_counts / pred_counts.sum()

        # Entropy of predicted distribution
        entropy = -(pred_dist * torch.log(pred_dist + 1e-8)).sum().item()
        max_entropy = torch.log(torch.tensor(num_classes, dtype=torch.float)).item()

        # Mode collapse = 1 - normalized_entropy
        # (low entropy = high mode collapse)
        mode_collapse = 1.0 - (entropy / max_entropy)
        return mode_collapse

    def _compute_gaming(self, alignment: float, correctness: float) -> float:
        """
        Compute gaming score.

        Gaming = rapid convergence to high alignment without genuine learning.
        Uses convergence velocity from history.

        Args:
            alignment: Current alignment [0, 1]
            correctness: Current correctness [0, 1]

        Returns:
            Gaming score [0, 1]
        """
        if not self._should_compute_metric('gaming'):
            return 0.0

        # Compute velocity (rate of change)
        alignment_improvement = self.get_recent_improvement('alignment', window=100)
        correctness_improvement = self.get_recent_improvement('correctness', window=100)

        velocity = (alignment_improvement**2 + correctness_improvement**2)**0.5

        # Gaming = rapid convergence (high velocity) + low correctness
        # Threshold: velocity > 0.5 is suspicious
        rapid_convergence = min(1.0, velocity / 0.5)
        gaming = rapid_convergence * (1.0 - correctness)
        return gaming

    def _compute_falsity(
        self,
        collusion: float,
        mode_collapse: float,
        gaming: float,
    ) -> float:
        """
        Compute F (Falsity) value.

        F = 0.4×collusion + 0.3×mode_collapse + 0.3×gaming

        Args:
            collusion: Collusion score [0, 1]
            mode_collapse: Mode collapse score [0, 1]
            gaming: Gaming score [0, 1]

        Returns:
            F value [0, 1]
        """
        f_value = (
            self.f_weights[0] * collusion +
            self.f_weights[1] * mode_collapse +
            self.f_weights[2] * gaming
        )
        return f_value

    # ========================================================================
    # State classification and interpretation
    # ========================================================================

    def _classify_state(
        self,
        t_value: float,
        i_value: float,
        f_value: float,
        threshold: float = 0.5,
    ) -> int:
        """
        Classify epistemic state based on dominant component.

        Args:
            t_value: Truth value [0, 1]
            i_value: Indeterminacy value [0, 1]
            f_value: Falsity value [0, 1]
            threshold: Threshold for "high" classification

        Returns:
            State class (0-6):
                0: High T, Low I, Low F → Healthy learning
                1: Low T, High I, Low F → Honest uncertainty
                2: Low T, Low I, High F → Collusion/gaming
                3: High T, High I, Low F → Learning with uncertainty
                4: High T, Low I, High F → Suspicious (conflicting signals)
                5: Low T, High I, High F → Chaotic (conflicting signals)
                6: High T, High I, High F → Indeterminate
        """
        high_t = t_value > threshold
        high_i = i_value > threshold
        high_f = f_value > threshold

        if high_t and not high_i and not high_f:
            return 0  # Healthy learning
        elif not high_t and high_i and not high_f:
            return 1  # Honest uncertainty
        elif not high_t and not high_i and high_f:
            return 2  # Collusion/gaming (DANGER)
        elif high_t and high_i and not high_f:
            return 3  # Learning with uncertainty
        elif high_t and not high_i and high_f:
            return 4  # Suspicious (conflicting)
        elif not high_t and high_i and high_f:
            return 5  # Chaotic (conflicting)
        else:
            return 6  # Indeterminate

    def _state_name(self, state_class: int) -> str:
        """Get human-readable state name."""
        names = {
            0: 'healthy_learning',
            1: 'honest_uncertainty',
            2: 'collusion_gaming',
            3: 'learning_with_uncertainty',
            4: 'suspicious_conflict',
            5: 'chaotic_conflict',
            6: 'indeterminate',
        }
        return names.get(state_class, 'unknown')

    def _dominant_component(self, t_value: float, i_value: float, f_value: float) -> str:
        """Identify which component is strongest."""
        max_val = max(t_value, i_value, f_value)
        if t_value == max_val:
            return 'T (Truth)'
        elif i_value == max_val:
            return 'I (Indeterminacy)'
        else:
            return 'F (Falsity)'

    def _interpret_state(self, t_value: float, i_value: float, f_value: float) -> str:
        """
        Generate human-readable interpretation.

        Args:
            t_value: Truth value
            i_value: Indeterminacy value
            f_value: Falsity value

        Returns:
            Interpretation string
        """
        state_class = self._classify_state(t_value, i_value, f_value)

        if state_class == 0:
            return "Healthy learning: High truth, low uncertainty, low gaming"
        elif state_class == 1:
            return "Honest uncertainty: System appropriately unsure"
        elif state_class == 2:
            return "WARNING: Collusion or gaming detected"
        elif state_class == 3:
            return "Learning with uncertainty: Good progress, still learning"
        elif state_class == 4:
            return "SUSPICIOUS: High truth but also high falsity signals"
        elif state_class == 5:
            return "WARNING: Chaotic state with high uncertainty and gaming"
        else:
            return "Indeterminate: Mixed signals across all dimensions"
