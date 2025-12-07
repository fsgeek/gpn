---
layout: distill
title: "The Coverage Boundary: Why High-Fidelity Primitives Don't Compose"
description: "A controlled experiment showing that adversarially trained primitives hit a glass ceiling on compositional generalization, while low-fidelity pedagogical primitives achieve perfect transfer."
date: 2026-04-28
future: true
htmlwidgets: true
authors:
  - name: Anonymous
tags: [compositionality, generalization, neural networks, representation learning]
---

## The Question

Can neural networks learn compositional rules that generalize beyond their training data?

The compositional generalization literature has established that models can succeed on *novel combinations of known primitives*: a system that learns "red circle" and "blue square" can often generate "red square" [Keysers et al., 2020; Park et al., 2021]. But this hides a critical assumption:

> **What exactly counts as a “known” primitive?**

Consider a generative model pre-trained to produce images of the digit 7. Does high visual fidelity—the ability to render a photorealistic 7—constitute “knowing” the digit well enough to use it compositionally in relational tasks like “generate X > Y”?

Intuitively, we assume better primitives yield better composition. If a model can generate a crisp, perfect digit, it must understand that digit.

**We found the opposite.**

In a controlled experiment, we show that high-fidelity primitives trained adversarially (GANs) hit a *glass ceiling* of composability, while low-fidelity "blotchy" primitives trained pedagogically achieve perfect transfer.

---

## Background: Coverage and Compositionality

Recent work has clarified that compositional generalization is constrained by the *coverage* of primitives in training data. Benchmarks like SCAN and its descendants show that models struggle on held-out combinations when key primitives never appear in the right structural contexts [Keysers et al., 2020]. The **Coverage Principle** formalizes this: for pattern-matching learners, reliable generalization is only possible within the "coverage" of functionally equivalent fragments seen during training [Chang et al., 2025]. In other words, coverage is a **necessary condition** for compositional generalization.

Our experiments take this as a starting point. We instantiate the Coverage Principle in an intentionally simple generative setting and then ask a deeper question: **even when coverage is satisfied, do all primitives admit compositional use?**

---

## The Experiment

To investigate this boundary, we designed a deliberately simple experiment using **Relational MNIST**. The task: generate three-digit displays of the form `[X][>][Y]` where `X` and `Y` are MNIST-style digits and `X > Y` numerically.

The simplicity is intentional. MNIST is the petri dish, not the ecology. If the coverage boundary failed to appear here—in the most controlled possible environment—it would suggest the phenomenon is an artifact of complexity. That it appears so sharply in this minimal setting implies a fundamental property of neural compositionality that scale may *mask* but cannot *cure*.

Our approach follows **pedagogical training with frozen primitives**. We pre-trained a *single-digit weaver* to generate individual digits `[0-9]`, then froze it and trained only a compositional layer—the *latent splitter*—to route latent codes for generating relational displays.

Crucially, we compared two types of teachers for the primitive generator:

1. **Adversarial (GAN):** Optimized to fool a discriminator, producing sharp, high-fidelity digits rich in texture.
2. **Pedagogical (Ours):** Optimized for structural reconstruction, producing abstract, low-frequency representations that preserve topology but discard texture.

![Figure 1: The Fidelity Trap. Left: Standard GAN samples—high contrast and sharp edges, but carrying pseudo-texture learned to satisfy an adversarial discriminator. Right: Our pedagogical samples—visually blotchy and diffuse, prioritizing topological clarity over textural noise.](assets/img/fig4_fidelity_comparison.png)

---

## Experimental Architecture & Controls

We separate *primitive competence* from *relational competence* by freezing the primitive generator and training only the relational layer.

- **Single-Digit Weaver (Frozen).** Pre-trained on digits `[0-9]` using either adversarial (GAN) or pedagogical objectives. Once trained, its weights are frozen.
- **Latent Splitter (Trainable).** Receives a latent code and learns to route it into `(X, >, Y)` displays, implementing relational structure over the same primitives.
- **Static Judge (Ground Truth Oracle).** Evaluates whether `X > Y` holds numerically. The judge is fixed and never trained.

Key controls:

- The primitive generator **never** sees the relational test set.
- The relational layer (latent splitter) is trained only on training relations; held-out relations are used purely for evaluation.
- After early experiments revealed collusion when the student could influence the teacher, we removed all student-to-teacher reward paths: the teacher’s objective depends solely on student performance as evaluated by the static judge.
- During all relational experiments, primitive generators are **frozen**, ensuring that differences in performance arise from the training objectives used to build primitives, not from additional fine-tuning.

![Figure 2: Experimental architecture. We freeze the primitive generator (whether GAN or Pedagogical) and train only the relational routing layer.](assets/img/fig3_experimental_design.png)

---

## The Fidelity Trap

A surprising observation emerged during primitive training. The pedagogical teacher generated digits that were visually *blotchy*—diffuse, soft-edged, and topologically abstract.

This apparent degradation was actually a feature. By removing texture (noise), the teacher forced the student to learn geometry (signal). The student could not "game" the metric by matching pixels; it had to learn the invariant structure of each digit.

In the discriminative setting, Geirhos et al. famously showed that ImageNet-trained CNNs are strongly biased toward texture, and that increasing shape bias improves robustness and generalization [Geirhos et al., 2019]. Our results suggest an analogous phenomenon on the generative side: adversarial objectives encourage texture-rich primitives that look good but compose poorly, whereas pedagogical objectives yield "blotchy" but **topological** primitives that compose perfectly.

This matters methodologically: because our primitives encode topology rather than texture, logical failures in the relational task cannot be attributed to pixel-level distribution shift. The model knew the abstract form of "7" perfectly. The only remaining question was whether it could *use* that knowledge compositionally.

---

## The Coverage Boundary

We first asked whether primitives could compose *without* specific relational training coverage.

- **Condition (Phase 1.5):**
  Train relational displays only for digits `[0-4]` (10 valid `X > Y` pairs).
  Test on relational displays for digits `[5-9]`, which are **completely unseen** in relational context.

- **Result:**
  **0% digit accuracy** and ~**chance-level relation accuracy** on the novel digits.

The model produced recognizable digits in isolation but garbage in relational contexts. This concretely instantiates the **Coverage Principle** [Chang et al., 2025] in a generative setting:

> **Primitive Competence** (being able to draw a 7)
> **does not grant**
> **Compositional License** (using 7 correctly in a relation).

As the Coverage Principle predicts, license is only acquired when a primitive appears in a *relational* context during training. Coverage is necessary—but, as we show next, it is not sufficient.

![Figure 3: The coverage boundary and glass ceiling. Same architecture, different training objectives, opposite outcomes.](assets/img/fig1_coverage_boundary.png)

---

## The Showdown: The Glass Ceiling of Adversarial Training

Once we established that coverage is necessary, we asked the deeper question:

> **Is coverage sufficient?**

If we give an adversarial model every advantage—full relational coverage, identical architecture, and visually superior primitives—can it match pedagogical performance?

We ran the experiment on **Novel Combinations** (Phase 1.6):

- **Training:**
  Digits `[0-9]`, with 41 of 45 valid `X > Y` pairs. We hold out four specific relations (e.g., `7 > 3`, `8 > 2`, `9 > 1`, `6 > 4`). Every digit appears in many relational contexts during training.

- **Testing:**
  Only the 4 held-out relations. These are *novel combinations of seen digits*.

**Results:**

| Training Objective | Primitives | Held-out Relation Accuracy |
|--------------------|------------|----------------------------|
| Pedagogical (Ours) | Blotchy    | **100.0%** |
| Adversarial (GAN)  | Crisp      | **81.1%** |

Similar symptoms have been reported at scale in text-to-image systems: models can render individual concepts with high fidelity yet catastrophically fail on compositional prompts (negation, counting, spatial relations), even when evaluation metrics like FID remain strong [Park et al., 2021; Huang et al., 2023; Vatsa et al., 2025]. These works document the **what**. Our result isolates a candidate **why**: adversarial objectives encourage entangled, texture-heavy representations that cannot be perfectly recomposed, even under full relational coverage.

The adversarial model is not "broken." 81% is not failure—it is a *ceiling*. The model had full relational coverage. It had seen every digit in compositional context. Yet it could not fully compose.

This is the **Glass Ceiling of Adversarial Training**. The model pays a **tax on composition**: capacity spent maintaining the illusion of texture leaves representations entangled in ways that resist perfect reassembly. No amount of additional coverage can break through, because the limitation is **geometric, not statistical**.

By contrast, our pedagogical primitives—although visually worse—are topologically clean and compose perfectly.

---

## Why It Matters: Contract of Appearance vs. Contract of Meaning

This experiment is a critique of how we train generative models.

Modern practice follows a **Contract of Appearance**. Adversarial objectives (GANs) and preference optimization (RLHF/RLAIF) teach models to *mimic the surface statistics* of correct answers. As our GAN results show, this produces high-fidelity primitives that look perfect to a critic but are hollow to a composer. They possess **Primitive Competence** but lack **Compositional License**. This same pattern appears in large language models trained with reinforcement learning from human feedback (RLHF/RLAIF): optimizing for human-rated plausibility can privilege surface agreement over structural understanding, with downstream costs to robustness and compositional generalization [Vatsa et al., 2025].

Our pedagogical approach enforces a **Contract of Meaning**. By using “blotchy,” abstract primitives, we deny the model the ability to succeed through texture mimicry. It must instead learn the *topology* of the digit—the “Platonic form”—because the texture is unavailable.

We hypothesize that **safety and generalization are the same goal**. A model that truly understands the structure of a concept (rather than just its likelihood) may be a model that can be trusted to handle that concept in novel contexts. If this holds, fixing the fragility of AI may require us to stop optimizing for how things *look* and start optimizing for how they *compose*.

Although demonstrated here on MNIST for clarity, the Coverage Boundary and Glass Ceiling are **architectural** phenomena, not dataset quirks. Large-scale generative training (GANs, diffusion, preference tuning) may be subject to the same fidelity trap: objectives that reward appearance can actively degrade compositional reasoning, even when coverage is abundant.

---

## Summary

| Experiment | What's Novel | Result |
|------------|--------------|--------|
| **Phase 1.5** | Novel Primitives (No Relational Coverage) | **0% Transfer** — The Coverage Boundary |
| **Phase 1.6** | Adversarial Primitives (Full Coverage)    | **81.1% Accuracy** — The Glass Ceiling |
| **Phase 1.6** | Pedagogical Primitives (Full Coverage)    | **100% Accuracy** — The Contract of Meaning |

The Coverage Boundary tells us *when* composition is possible.
The Glass Ceiling tells us *whether* the primitives are capable of it.

You need both: primitives shaped for meaning, and coverage that licenses their use.

Code and experimental details will be released upon acceptance.

---

## Related Work

Our setup connects to several strands of prior work. Compositional generalization benchmarks such as SCAN and its extensions highlight the importance of primitive coverage in sequence-to-sequence models [Keysers et al., 2020; Oren et al., 2022]. The **Coverage Principle** of Chang et al. (2025) formalizes coverage as a necessary condition for pattern-matching learners, a condition our "Coverage Boundary" experiment instantiates in a generative regime [Chang et al., 2025].

On the generative side, compositional text-to-image benchmarks repeatedly find that models with excellent perceptual quality metrics still fail on novel combinations of attributes and objects [Park et al., 2021; Huang et al., 2023]. Vatsa et al. (2025) describe this as "right looks, wrong reasons," emphasizing failures of compositional fidelity in modern diffusion models [Vatsa et al., 2025]. Our **Glass Ceiling** result pinpoints adversarial objectives as one mechanism that can produce this pattern, even in a minimal MNIST petri dish.

Finally, our "blotchy but topological" primitives resonate with work on shape vs. texture bias in CNNs [Geirhos et al., 2019] and with emerging views of deep representations as learning topological manifolds amenable to symbolic or relational reuse. We view our pedagogical objective as a small, controlled example of **training for meaning rather than appearance**—a design choice that may scale to more realistic architectures and datasets.

---

## Limitations & Next Steps

**Toy domain.** Our experiments use MNIST to make the phenomenon as visible and controllable as possible. Real-world data are higher dimensional and noisier, but if the fidelity trap appears in this simplest setting, we expect it to persist—if hidden—at scale.

**Frozen primitives.** We freeze the digit generator when training relations to cleanly separate primitive learning from relational learning. Future work could study joint training and analyze how much compositional capacity can be recovered—or destroyed—when primitives continue to adapt.

**Single relation.** We focus on a single relational operator (`>`). Extending to multiple relations (equality, ordering, arithmetic expressions) and to symbolic domains would test whether pedagogical primitives systematically support richer compositional logics.

**Beyond MNIST.** The natural next step is to apply pedagogical objectives to more complex visual and language domains, and to compare them directly against adversarial or preference-based objectives used in modern AI training pipelines.

If the fidelity trap generalizes, then **training models to teach rather than to mimic** may be a necessary ingredient in building systems that truly understand—and safely extend—what they learn.
