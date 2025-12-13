"""
Deterministic Judge for SCAN-lite.

Unlike the image Judge (neural classifier), this Judge is rule-based.
Given an action sequence, it can:
1. Determine if the sequence is valid
2. Parse it back to the original command
3. Check correctness against expected output

This provides ground truth for pedagogical training.

Extended to handle AMBIGUOUS examples with multiple valid outputs.
For ambiguous inputs, correctness = True if output matches ANY valid interpretation.
"""

import torch
from typing import Optional, Tuple, List, Set
from src.data.scan_lite import (
    ACTIONS, MODIFIERS, COUNTS,
    OUTPUT_ACTIONS, OUTPUT_TURNS,
    ACTION_TO_IDX, IDX_TO_ACTION,
    COMMAND_TO_IDX,
    PAD_TOKEN, SOS_TOKEN, EOS_TOKEN,
    generate_action_sequence,
    detokenize_actions,
    SCANExample,
)


class SCANJudge:
    """
    Rule-based Judge for SCAN-lite sequences.

    Evaluates generated action sequences against ground truth.
    """

    def __init__(self):
        # Reverse mappings for parsing
        self.action_to_primitive = {v: k for k, v in OUTPUT_ACTIONS.items()}
        self.turn_to_modifier = {v: k for k, v in OUTPUT_TURNS.items()}

        # Build expected outputs for all commands
        self.expected_outputs = {}
        for action in ACTIONS:
            for modifier in [None] + MODIFIERS:
                for count in [None] + COUNTS:
                    cmd_parts = [action]
                    if modifier:
                        cmd_parts.append(modifier)
                    if count:
                        cmd_parts.append(count)
                    cmd = ' '.join(cmd_parts)
                    expected = generate_action_sequence(action, modifier, count)
                    self.expected_outputs[expected] = cmd

    def evaluate_sequence(
        self,
        generated: str,
        expected: str,
    ) -> Tuple[bool, float]:
        """
        Evaluate a generated sequence against expected.

        Args:
            generated: Generated action sequence string
            expected: Expected action sequence string

        Returns:
            (is_correct, token_accuracy)
        """
        gen_tokens = generated.split()
        exp_tokens = expected.split()

        is_correct = (generated == expected)

        # Token accuracy
        max_len = max(len(gen_tokens), len(exp_tokens))
        if max_len == 0:
            token_accuracy = 1.0 if is_correct else 0.0
        else:
            matches = sum(1 for g, e in zip(gen_tokens, exp_tokens) if g == e)
            token_accuracy = matches / max_len

        return is_correct, token_accuracy

    def evaluate_batch(
        self,
        generated_indices: torch.Tensor,
        expected_indices: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Evaluate a batch of generated sequences.

        Args:
            generated_indices: (batch, seq_len) generated token indices
            expected_indices: (batch, seq_len) expected token indices

        Returns:
            correct: (batch,) boolean tensor of correctness
            token_acc: (batch,) token accuracy per example
        """
        batch_size = generated_indices.size(0)
        correct = []
        token_acc = []

        for i in range(batch_size):
            gen_str = detokenize_actions(generated_indices[i].tolist())
            exp_str = detokenize_actions(expected_indices[i].tolist())

            is_correct, acc = self.evaluate_sequence(gen_str, exp_str)
            correct.append(is_correct)
            token_acc.append(acc)

        return (
            torch.tensor(correct, dtype=torch.bool, device=generated_indices.device),
            torch.tensor(token_acc, dtype=torch.float, device=generated_indices.device),
        )

    def get_correctness_signal(
        self,
        generated_indices: torch.Tensor,
        expected_indices: torch.Tensor,
    ) -> torch.Tensor:
        """
        Get correctness signal for training.

        Returns soft correctness score (token accuracy) suitable for loss computation.

        Args:
            generated_indices: (batch, seq_len) generated token indices
            expected_indices: (batch, seq_len) expected token indices

        Returns:
            correctness: (batch,) float tensor in [0, 1]
        """
        _, token_acc = self.evaluate_batch(generated_indices, expected_indices)
        return token_acc

    def parse_to_command(self, action_sequence: str) -> Optional[str]:
        """
        Parse an action sequence back to its command.

        Args:
            action_sequence: Action sequence string (e.g., "LTURN WALK LTURN WALK")

        Returns:
            Command string if parseable, None otherwise
        """
        return self.expected_outputs.get(action_sequence)

    def is_valid_sequence(self, action_sequence: str) -> bool:
        """Check if an action sequence corresponds to any valid command."""
        return action_sequence in self.expected_outputs

    def evaluate_against_set(
        self,
        generated: str,
        valid_outputs: List[str],
    ) -> Tuple[bool, float, int]:
        """
        Evaluate a generated sequence against a SET of valid outputs.

        For ambiguous examples, ANY match is correct.

        Args:
            generated: Generated action sequence string
            valid_outputs: List of valid action sequences

        Returns:
            (is_correct, best_token_accuracy, matched_interpretation_idx)
            matched_interpretation_idx is -1 if no match
        """
        best_acc = 0.0
        matched_idx = -1

        for i, valid in enumerate(valid_outputs):
            is_match, acc = self.evaluate_sequence(generated, valid)
            if is_match:
                return True, 1.0, i
            if acc > best_acc:
                best_acc = acc
                matched_idx = i

        return False, best_acc, matched_idx

    def evaluate_ambiguous_batch(
        self,
        generated_indices: torch.Tensor,
        examples: List[SCANExample],
    ) -> Tuple[torch.Tensor, torch.Tensor, List[int]]:
        """
        Evaluate a batch where some examples may be ambiguous.

        Args:
            generated_indices: (batch, seq_len) generated token indices
            examples: List of SCANExample (may have valid_outputs set for ambiguous)

        Returns:
            correct: (batch,) boolean tensor
            token_acc: (batch,) best token accuracy
            matched_interpretations: list of which interpretation matched (-1 if none)
        """
        batch_size = generated_indices.size(0)
        correct = []
        token_acc = []
        matched_interps = []

        for i in range(batch_size):
            gen_str = detokenize_actions(generated_indices[i].tolist())
            ex = examples[i]

            if ex.is_ambiguous and ex.valid_outputs:
                # Ambiguous: check against all valid outputs
                is_correct, acc, matched = self.evaluate_against_set(gen_str, ex.valid_outputs)
            else:
                # Unambiguous: single expected output
                is_correct, acc = self.evaluate_sequence(gen_str, ex.action_sequence)
                matched = 0 if is_correct else -1

            correct.append(is_correct)
            token_acc.append(acc)
            matched_interps.append(matched)

        return (
            torch.tensor(correct, dtype=torch.bool, device=generated_indices.device),
            torch.tensor(token_acc, dtype=torch.float, device=generated_indices.device),
            matched_interps,
        )


class SCANJudgeWrapper:
    """
    Wrapper to make SCANJudge interface compatible with GPN training.

    Provides forward() method that returns logits-like output for grounding loss.
    """

    def __init__(self, num_commands: int = 64):
        self.judge = SCANJudge()
        self.num_commands = num_commands

        # Map commands to indices
        self.command_to_idx = {}
        idx = 0
        for action in ACTIONS:
            for modifier in [None] + MODIFIERS:
                for count in [None] + COUNTS:
                    cmd_parts = [action]
                    if modifier:
                        cmd_parts.append(modifier)
                    if count:
                        cmd_parts.append(count)
                    cmd = ' '.join(cmd_parts)
                    self.command_to_idx[cmd] = idx
                    idx += 1

    def forward(
        self,
        generated_indices: torch.Tensor,
        expected_indices: torch.Tensor,
    ) -> torch.Tensor:
        """
        Get Judge signal for training.

        Returns correctness as a signal (not logits, since Judge is deterministic).

        Args:
            generated_indices: (batch, seq_len) generated token indices
            expected_indices: (batch, seq_len) expected token indices

        Returns:
            correctness: (batch,) correctness score in [0, 1]
        """
        return self.judge.get_correctness_signal(generated_indices, expected_indices)

    def evaluate(
        self,
        generated_indices: torch.Tensor,
        expected_indices: torch.Tensor,
    ) -> Tuple[float, float]:
        """
        Evaluate batch and return aggregate metrics.

        Returns:
            (sequence_accuracy, token_accuracy) averaged over batch
        """
        correct, token_acc = self.judge.evaluate_batch(generated_indices, expected_indices)
        return correct.float().mean().item(), token_acc.mean().item()

    def evaluate_with_examples(
        self,
        generated_indices: torch.Tensor,
        examples: List[SCANExample],
    ) -> Tuple[float, float, List[int]]:
        """
        Evaluate batch using SCANExample objects (supports ambiguous examples).

        Args:
            generated_indices: (batch, seq_len) generated token indices
            examples: List of SCANExample (may include ambiguous examples)

        Returns:
            (sequence_accuracy, token_accuracy, matched_interpretations)
        """
        correct, token_acc, matched = self.judge.evaluate_ambiguous_batch(
            generated_indices, examples
        )
        return correct.float().mean().item(), token_acc.mean().item(), matched


def create_scan_judge() -> SCANJudgeWrapper:
    """Factory function for SCAN Judge."""
    return SCANJudgeWrapper()


if __name__ == '__main__':
    print("SCANJudge Test")
    print("=" * 50)

    judge = create_scan_judge()

    # Test evaluation
    gen_seq = "LTURN WALK LTURN WALK"  # "walk left twice"
    exp_seq = "LTURN WALK LTURN WALK"

    is_correct, acc = judge.judge.evaluate_sequence(gen_seq, exp_seq)
    print(f"'{gen_seq}' vs '{exp_seq}'")
    print(f"  Correct: {is_correct}, Token acc: {acc:.2f}")

    # Test incorrect
    gen_seq = "LTURN WALK LTURN RUN"  # Wrong!
    is_correct, acc = judge.judge.evaluate_sequence(gen_seq, exp_seq)
    print(f"'{gen_seq}' vs '{exp_seq}'")
    print(f"  Correct: {is_correct}, Token acc: {acc:.2f}")

    # Test parsing
    seq = "LTURN JUMP LTURN JUMP LTURN JUMP LTURN JUMP"
    cmd = judge.judge.parse_to_command(seq)
    print(f"\nParse '{seq}'")
    print(f"  -> '{cmd}'")

    # Test with tensors
    from src.data.scan_lite import tokenize_actions

    gen_tokens = torch.tensor([tokenize_actions("LTURN WALK LTURN WALK")])
    exp_tokens = torch.tensor([tokenize_actions("LTURN WALK LTURN WALK")])

    seq_acc, tok_acc = judge.evaluate(gen_tokens, exp_tokens)
    print(f"\nTensor evaluation: seq_acc={seq_acc:.2f}, tok_acc={tok_acc:.2f}")

    # Test ambiguous example evaluation
    print("\n" + "=" * 50)
    print("AMBIGUOUS EXAMPLE EVALUATION")
    print("=" * 50)

    from src.data.scan_lite import get_ambiguous_examples

    ambiguous = get_ambiguous_examples()

    # Test: "walk and run left" with both valid interpretations
    ex = ambiguous[0]  # "walk and run left"
    print(f"\nCommand: '{ex.command}'")
    print(f"Valid outputs: {ex.valid_outputs}")

    # Test with interpretation 1
    gen_seq = "WALK LTURN RUN"  # Tight binding
    is_correct, acc, matched = judge.judge.evaluate_against_set(gen_seq, ex.valid_outputs)
    print(f"\nGenerated: '{gen_seq}'")
    print(f"  Correct: {is_correct}, Best acc: {acc:.2f}, Matched interpretation: {matched}")

    # Test with interpretation 2
    gen_seq = "LTURN WALK LTURN RUN"  # Distributed
    is_correct, acc, matched = judge.judge.evaluate_against_set(gen_seq, ex.valid_outputs)
    print(f"Generated: '{gen_seq}'")
    print(f"  Correct: {is_correct}, Best acc: {acc:.2f}, Matched interpretation: {matched}")

    # Test with wrong output
    gen_seq = "WALK WALK WALK"  # Wrong!
    is_correct, acc, matched = judge.judge.evaluate_against_set(gen_seq, ex.valid_outputs)
    print(f"Generated: '{gen_seq}'")
    print(f"  Correct: {is_correct}, Best acc: {acc:.2f}, Matched interpretation: {matched}")
