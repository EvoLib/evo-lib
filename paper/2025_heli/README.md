# HELI Preprint

This directory contains the preprint for:

> **HELI: Stabilising Structural Mutation through Lineage Incubation**

Archived on Zenodo:

DOI: 10.5281/zenodo.18045147

---

## Overview

HELI (Hierarchical Evolution with Lineage Incubation) is a neuroevolution mechanism designed to stabilize structural mutations such as:

- recurrent connections,
- topology growth,
- connectivity changes,
- and connectivity restructuring.

The core idea is simple:

> structurally mutated individuals are temporarily isolated in short-lived lineage populations before re-entering global competition.

This incubation period allows unstable topological innovations to adapt locally before facing immediate selection pressure.

HELI is motivated by a central problem in topology-evolving neuroevolution systems:

> useful structural mutations are often eliminated before they can become functional.

The approach is conceptually related to:

- NEAT speciation,
- ALPS,
- Novelty Search,
- Quality-Diversity methods,
- and developmental buffering in evolutionary biology.

---

## Main Contribution

HELI proposes that:

> immediate global selection pressure can suppress structural innovation.

Instead of evaluating structural mutations instantly, HELI introduces a protected maturation phase for architectural adaptation.

---

## Experimental Scope

The paper evaluates HELI on small deterministic benchmark tasks including:

- Parity-3 classification
- Delay-5 sequential prediction
- Sine-wave regression

The strongest results appear in recurrent memory tasks where recurrent structures must emerge through structural mutation rather than being predefined.

---

## Why It Matters

Many topology-evolving neuroevolution systems struggle with structural mutations because newly introduced architectures are often temporarily disruptive.

HELI directly targets this instability by introducing a protected incubation phase for structural innovation.

The mechanism may be particularly relevant for problems where useful structures emerge gradually, such as:

- recurrent connections,
- memory systems,
- modular organization,
- and adaptive topology growth.

While the current experiments are intentionally small-scale, the results suggest that temporary developmental protection can substantially improve the survival of structural mutations.

---

## Computational Tradeoff

The incubation process introduces additional evaluations, trading computational efficiency for increased structural stability.

The paper addresses this through evaluation-budget normalization, but scalability and real-world wall-clock efficiency remain open questions.
