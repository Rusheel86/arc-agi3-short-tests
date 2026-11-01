# Third-party components

Our own code in this folder is MIT-0 (`LICENSE`). Everything below was written by others. It keeps its own licence, and
the MIT-0 grant does not cover it. The Apache-2.0 parts shipped here (the K8V4 port, and inside the notebook Franzen's
patch and base cells and Lord Han Solo's adapted serving code) come with the licence text (`LICENSES/Apache-2.0.txt`)
and a `NOTICE` file that keeps their notices and states our changes. "How sure" says where the licence statement comes from:

- **verified**: read at the source on 2026-10-28 (a Kaggle page or field, a GitHub LICENSE file or API answer, a model card);
- **local copy**: read in the copy we fetched (2026-10-08) or in the file itself;
- **unconfirmed**: the sources disagree or say nothing, so we cannot say which licence applies.

This is a reading of the licence texts, not legal advice.

## What the notebook inherits

| Component | Author | Licence | How sure | Source | How it enters |
|---|---|---|---|---|---|
| Duck harness and TAAF framework (`ARC3-Inference`, `tufa-arc-agi-framework`), commit 7652836 | Jeroen Cottaar, Tufa Labs | **unconfirmed**: the Kaggle notebook `jeroencottaar/tufa-labs-duck-harness-june-30-milestone-winner` says Apache 2.0; the Kaggle dataset `jeroencottaar/taaf-kaggle-source-share` says MIT; `ARC3-Inference/pyproject.toml` has the MIT classifier; the GitHub repository `Tufalabs/duck-harness` has no LICENSE file (the GitHub API reports no licence) | verified (each statement); which applies: unconfirmed | kaggle.com/code/jeroencottaar/tufa-labs-duck-harness-june-30-milestone-winner; github.com/Tufalabs/duck-harness | Kaggle dataset `dfranzen/taaf-kaggle-source-bundle-copy` (an unchanged copy; its licence field says "unknown") and `lordhansolo/taaf-kaggle-source` (licence field Apache-2.0). Not copied into this folder |
| Milestone 2 solution: the harness patch (harness, prompts, scheduler), the notebook's base cells and its "About this notebook" text, launch settings | Daniel Franzen | Apache-2.0 | verified (Kaggle notebook page; the LICENSE file of github.com/da-fr/arc-agi-3-solution, whose copyright line is the unfilled template) | kaggle.com/code/dfranzen/arc-agi-3-milestone-2-solution | The `harness-changes.patch` cell and the base cells of `notebook/arc3-e154-keepalive.ipynb`. `patches/apply_patches.py` reads that patch from the notebook; it is not duplicated in `patches/` |
| vLLM serving stack: vLLM main e975732 with an overlay, the mixed NVFP4/FP8 launcher and watchdog, the 32k draft vocabulary | Lord Han Solo; the vLLM project | Apache-2.0 | verified (Kaggle notebook page; dataset licence fields; vLLM's GitHub licence) | kaggle.com/code/lordhansolo/built-on-tufa-labs-duck-harness-milestone-2; datasets `lordhansolo/vllm-main-e975732-arc3`, `lordhansolo/taaf-kaggle-source`; github.com/vllm-project/vllm | Datasets and the notebook attached as inputs; the notebook's server cell adapts his launcher (our changes in it are E-marked) |
| K8V4 KV-cache port (`k8v4_port/e083_k8v4_port.py`; the notebook's E073 prepare cell writes the same bytes, the server cell's launcher applies it) | nvfp4 bit helpers and value quantiser by andreasgru (`vllm-qwen38-flash-next-kv-quant`, `kvq-patch.py`, after vLLM pull requests 44389 and 54846); the mixed fp8-key / nvfp4-value layout, its writer and the kernel's value-only branch are ours | Apache-2.0 for the whole file, as its own header states (it contains andreasgru's code) | local copy (andreasgru's LICENSE file is the Apache License 2.0; the port's header) | the port's header names the source | Shipped as `k8v4_port/` (the file byte-identical to the notebook's sha256-pinned copy, its upstream `LICENSE`, a README) and embedded in the notebook; everything it inserts into vLLM carries the marker `ARC3-KVQ-E083`. Not MIT-0. Apache-2.0 text: `LICENSES/Apache-2.0.txt`; notices and changes: `NOTICE` |
| Pennyroyal SGLang wheelhouse v2.5.3 | John Pezzulli (a fork of SGLang), with patches by Mamy Ratsimbazafy, Gabriel Olympie and Daniel Franzen | Apache-2.0 (SGLang and the Pennyroyal fork) | verified for the GitHub repositories; **unconfirmed** for the Kaggle dataset, whose licence field says "unknown" and whose description says bundled packages keep their upstream licences | github.com/jpezzulli/sglang-rtxpro6000; Kaggle dataset `dfranzen/pennyroyal-v253` | Dataset attached as an input (E098/E154 serve with vLLM; the input is kept from the inherited notebook) |
| ARC-AGI-3 competition data and wheels (`arc-prize-2026-arc-agi-3`) | ARC Prize Foundation | "Data access and use: Apache 2.0" (competition rules), plus the rules' data-security clause | verified | kaggle.com/competitions/arc-prize-2026-arc-agi-3 | Competition input; not redistributed |
| Python packages in the Kaggle image and the competition wheelhouse | various | each package's own licence | not checked one by one | Kaggle Docker image pinned in `notebook/kernel-metadata.json` | Installed by Kaggle, not redistributed |

## Model weights (not in this folder)

| Model | Author | Licence | How sure | How it enters |
|---|---|---|---|---|
| Qwen3.8-Flash-Next | Qwen team (Alibaba) | Qwen Community License 1.0, Copyright (c) 2026 Qwen; not OSI-approved | verified (huggingface.co/Qwen/Qwen3.8-Flash-Next, LICENSE file and card header `qwen-community-1.0`) | through the three builds below |
| mixed NVFP4/FP8 build with an MTP head (primitive-ai), mirrored as `lordhansolo/qwen3-8-flash-next-mixed-nvfp4-fp8` (PyTorch, hf-mixed-mtp-nvfp4, version 1) | primitive-ai; mirrored by Lord Han Solo | Qwen Community License 1.0 (the card repeats it) | verified | Kaggle Model input; the served model |
| Intel W4A16 AutoRound build, mirrored as `dfranzen/intel-qwen3.8-flash-next-w4a16-autoround` | Intel; mirrored by Daniel Franzen | Qwen Community License 1.0 (the card repeats it) | verified | Kaggle Model input (kept from the inherited notebook) |
| albucino's MTP draft head, mirrored as `dfranzen/albucino-qwen3-8-flash-next-drafter` | albucino; mirrored by Daniel Franzen | Qwen Community License 1.0 (the card repeats it) | verified | Kaggle Model input (kept from the inherited notebook) |

What the Qwen Community License 1.0 allows, in short: anyone may use, copy, modify, distribute and fine-tune the weights,
free of charge, keeping the copyright and licence notice. A product with more than 100,000,000 monthly users or
US$20,000,000 monthly revenue must display the model name, and a business offering Model-as-a-Service or an AI coding or
office assistant needs a separate licence from Qwen. This entry is a research notebook with none of these uses, and it
redistributes no weights: they stay Kaggle Model inputs.

**Open question.** The Kaggle rules' winner-licence clause asks for an open-source system, model and weights, with an
exception for "pretrained models with an incompatible license". Whether weights under the Qwen Community License meet
that wording has not been answered by the organisers (Kaggle forum threads 745079 and 745837; Kaggle staff said on
2026-10-16 they cannot interpret model licences).

## What the tools carry

| Item | Origin | Note |
|---|---|---|
| `tools/*/data/*benchmark.json`, `tools/e155_truncation/data/e154_smoke/` | our own runs of the agent on public ARC-AGI-3 games (E058, E061 screens; the E154 commit smoke) | slim copies: game id, level counts, the agent's actions per level, generated tokens per action, and (for the scoring rule) the public games' human baseline action counts per level, which come from the public game files (competition data, Apache 2.0 per the rules). No game frames, prompts or model text |
| `tools/paper_figures/data/agi2/` | our own ARC-AGI-2 experiment summaries (E142, E150-E152) | counts and public-evaluation task ids only |
| `tools/paper_figures/game_curve.py`, `tools/e155_truncation/game_curve.py` | ours | the E059/E072 truncation rule |
| colour palette in the figure scripts | a reference palette (colour values only, no code) | |
| numpy, scipy, matplotlib | their projects (BSD-style licences) | dependencies, not shipped |

## Not redistributed here

- The community-game datasets used by E110 and E155 (DreamTeam games: Apache-2.0 with an NVIDIA copyright line;
  ARC-Interactive games: MIT). `tools/e155_truncation` can read local copies of them; none is included.
- The E110 results archive (`e110_results.zip`, 0.67 GB, a Kaggle notebook output) that E155 analyses.
- For ARC-AGI-2 (not part of this release): NVARC's code and data and the BARC seeds, which state no licence.
