---
layout: distill
title: "The Coverage Boundary: Why High-Fidelity Primitives Don't Compose"
description: "A controlled experiment demonstrating that adversarial training imposes a 'glass ceiling' on compositionality. While pedagogical training achieves 100% transfer with 'blotchy' primitives, adversarial training caps at 81% despite full coverage."
date: 2026-04-28
future: true
htmlwidgets: true
authors:
  - name: Anonymous
tags: [compositionality, generalization, neural networks, representation learning]
---

## The Question

Can neural networks learn compositional rules that generalize beyond their training data? The answer depends critically on the training objective used to build their elementary concepts.

The compositional generalization literature has established that neural networks can succeed on novel combinations of known primitives. A model that learns "red circle" and "blue square" can often generate "red square." But what counts as a "known" primitive?

Consider a generative model pre-trained to produce images of the digit 7. Does high visual fidelity—the ability to render a photorealistic 7—constitute "knowing" the digit well enough to use it compositionally in relational tasks like "generate X > Y"?

Intuitively, we assume better primitives yield better composition. We assume that if a model can generate a crisp, perfect digit, it must understand that digit. **We found the opposite.** In a controlled experiment, we show that high-fidelity primitives trained adversarially (GANs) hit a "glass ceiling" of composability, while low-fidelity "blotchy" primitives trained pedagogically achieve perfect transfer.

## The Experiment

To investigate this boundary, we designed a deliberately simple experiment using Relational MNIST. The task: generate three-digit displays of the form [X][>][Y] where X and Y are MNIST-style digits and X > Y numerically.

The simplicity is intentional. MNIST is the petri dish, not the ecology. If the coverage boundary fails to appear here—in the most controlled possible environment—it would suggest the phenomenon is an artifact of complexity. That it appears so sharply in this minimal setting implies a fundamental property of neural compositionality that scale may mask but cannot cure.

Our approach follows **pedagogical training with frozen primitives**. We pre-trained a "single-digit weaver" to generate individual digits [0-9], then froze it and trained only a compositional layer—the "latent splitter"—to route latent codes for generating relational displays.

Crucially, we compared two types of teachers for the primitive generator:
1.  **Adversarial (GAN):** Optimized to fool a discriminator, producing sharp, high-fidelity digits rich in texture.
2.  **Pedagogical (Ours):** Optimized for structural reconstruction, producing abstract, low-frequency representations.

![Figure 1: The Fidelity Trap. Left: Standard GAN samples—high contrast and sharp edges, but carrying 'pseudo-texture' learned to satisfy an adversarial discriminator. Right: Our pedagogical samples—visually 'blotchy' and diffuse, prioritizing topological clarity over textural noise.](assets/img/fig4_fidelity_comparison.png)

## The Fidelity Trap

A surprising observation emerged during training. The pedagogical teacher generated primitives that were visually "blotchy"—diffuse, soft-edged, and topologically abstract.

This apparent degradation was actually a feature. By removing texture (noise), the teacher forced the student to learn geometry (signal). The student couldn't "game" the metric by matching pixels; it had to learn the invariant structure of each digit.

This matters methodologically: because our primitives encode topology rather than texture, logical failures cannot be attributed to pixel-level distribution shift. The model knew the abstract form of "7" perfectly.

![Figure 2: Experimental architecture. We freeze the primitive generator (whether GAN or Pedagogical) and train only the relational routing layer.](assets/img/fig3_experimental_design.png)


## The Finding: The Coverage Boundary

We first tested whether primitives could compose *without* specific relational training coverage.
* **Condition:** Train on relations for digits [0-4]. Test on relations for [5-9].
* **Result:** 0% Transfer.

The model produced recognizable digits in isolation but garbage in relational contexts. This establishes the **Coverage Boundary**: Primitive Competence (drawing a 7) does not grant Compositional License (using a 7 in a relation). License is only acquired when a primitive appears in a relational context during training.

## The Showdown: The Glass Ceiling

Once we established that coverage is necessary, we asked the deeper question: **Is coverage sufficient?**

If we give an adversarial model every advantage—full coverage, identical architecture, and visually superior primitives—can it match pedagogical performance?

We ran the experiment on "Novel Combinations." We trained on nearly all valid pairs (41 of 45), holding out 4 specific combinations (e.g., "7>3"). Both models had seen every digit in diverse relational contexts.

**The Result:**

| Training Objective | Primitives | Held-out Relation Accuracy |
|--------------------|------------|----------------------------|
| Pedagogical (Ours) | Blotchy | **100.0%** |
| Adversarial (GAN) | Crisp | **81.1%** |

The adversarial model wasn't broken. 81% is not failure—it's a ceiling. The model had full relational coverage. It had seen every digit in compositional context. Yet it could not fully compose.

This is the **Glass Ceiling of Adversarial Training**. The model pays a **tax on composition**—capacity spent maintaining the illusion of texture leaves representations entangled in ways that resist perfect reassembly. No amount of additional coverage can break through, because the limitation is geometric, not statistical.

![Figure 3: The coverage boundary and glass ceiling. Same architecture, different training objectives, opposite outcomes.](assets/img/fig1_coverage_boundary.png)

## Why It Matters: Contract of Appearance vs. Contract of Meaning

This experiment is a critique of how we train generative models.

We currently train under a **Contract of Appearance**. Adversarial objectives (like GANs) and preference optimization (like RLHF) teach models to *mimic the surface statistics* of a correct answer. As our GAN results show, this produces high-fidelity primitives that look perfect to a critic but are hollow to a composer. They possess **Primitive Competence** but lack **Compositional License**.

Our pedagogical approach enforces a **Contract of Meaning**. By using "blotchy," abstract primitives, we denied the model the ability to mimic. We forced it to learn the topology of the digit—the "Platonic form"—because the texture was unavailable.

We hypothesize that safety and generalization are the same goal. A model that truly understands the structure of a concept (rather than just its likelihood) may be a model that can be trusted to handle it in novel contexts. If this holds, fixing the fragility of AI may require us to stop optimizing for how things *look* and start optimizing for how they *compose*.

## Summary

| Experiment | What's Novel | Result |
|------------|--------------|--------|
| **Phase 1.5** | Novel Primitives (No Coverage) | **0% Transfer** (The Coverage Boundary) |
| **Phase 1.6** | Adversarial Primitives (Full Coverage) | **81.1% Accuracy** (The Glass Ceiling) |
| **Phase 1.6** | Pedagogical Primitives (Full Coverage) | **100% Accuracy** (The Contract of Meaning) |

The coverage boundary tells us *when* composition is possible. The glass ceiling tells us *if* the primitives are capable of it. You need both: primitives shaped for meaning, and coverage that licenses their use.

Code and experimental details available at: [repository link upon acceptance]
