Motion-context temporal anchor / payload patch logic in this build is adapted
from NikoDemon80/ComfyUI-H3-Motion-Context:
https://github.com/NikoDemon80/ComfyUI-H3-Motion-Context

Upstream license: GPL-3.0.
The adaptation removes disk Save/Load and connects the previous sampled H3 latent
directly in RAM inside one ComfyUI DAG.

This notice describes the adapted motion-context layer. The Master Extender
also has its own disk-backed clip cache and final decoding path.

## Extender foundation

Original project: tritant/ComfyUI_MiniMax_H3_Extender
https://github.com/tritant/ComfyUI_MiniMax_H3_Extender

This custom build includes adapted extender, motion-context, and decode code.
The supplied source snapshot carried an Apache-2.0 LICENSE, preserved verbatim
at licenses/Apache-2.0.txt. The motion-context files additionally identify their
GPL-3.0 origin above. The combined distribution is GPL-3.0; this does not erase
the original Apache-2.0 notices or change the license of separately distributed
upstream projects.

## Master Extender build (fork origin)

Original project: only2uuuu-hub/ComfyUI-MiniMax-H3-Master-Extender-Custom-built-with-Astra-6-
https://github.com/only2uuuu-hub/ComfyUI-MiniMax-H3-Master-Extender-Custom-built-with-Astra-6-

This repository (ganarajpr/comfyui-h3-master-extender) was created as a fork of
that build. The build's commits 9219cce, 086b4a4 and e72acce (2026-09-11) are
the base of its history, and its license (GPL-3.0) and notices are kept. On
2026-10-23 the repository was detached from the fork network. Every later
change is recorded in CHANGELOG.md and the git history.

## Separately installed dependencies

- LBH-123-AI/Comfyui_Minimax_h3_latent_Upscaler:
  https://github.com/LBH-123-AI/Comfyui_Minimax_h3_latent_Upscaler
  Provides MinimaxH3LatentUpscaler3D. Refer to that project for its terms.
  (The original build linked xmarre's copy, since renamed
  xmarre/Comfyui_Minimax_h3_latent_Upscaler-Plus.)
- pytraveler/MiniMax-H3-Prompt-Rewriter-ComfyUI:
  https://github.com/pytraveler/MiniMax-H3-Prompt-Rewriter-ComfyUI
  Model sessions for the built-in prompt rewriter and story engines.
- Comfy-Org/ComfyUI:
  https://github.com/Comfy-Org/ComfyUI
  Provides the runtime and native MiniMax H3 model integration.

Dependency code and model artifacts are not bundled in this release.

## Changes in this distribution

The Master Extender combines two-pass Turbo LoRA / HyperFlow Flash generation and learned 3D latent
upscaling with clip review, continuity, project controls, and final decoding.
The release prepared on 2026-09-11 blanks the initial Python and frontend prompt
defaults and supplies a sanitized workflow, requirements, and installation
documentation. Original author references in the source are retained.

The GPL-3.0 license text in LICENSE was copied from the local ComfyUI source
distribution. The prior Apache-2.0 license text is retained under licenses/.

## CodecOPSD codec gazer (reference-token pruning)

codecopsd_vendor/ is the codec gazer from CodecOPSD (Boyang Wang et al., arXiv:2611.31184,
https://github.com/Kiteretsu77/CodecOPSD), taken via the ComfyUI port
aaalll12322/ComfyUI-MiniMaxH3-CodecPrune (MIT, da078fe) which vendors it with one Windows
temp-file fix. Copyright (c) 2026 The CodecOPSD Authors; the file headers are kept. Used by
codec_prune.py for the optional ref_codec_prune input. The H.264 reader binary
(bin/codec_reader.exe) is built separately and is not part of this repository.
