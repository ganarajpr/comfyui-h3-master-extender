"""CodecOPSD codec pruning, vendored.

Source: https://github.com/Kiteretsu77/CodecOPSD  (src/codecopsd/codec_gazer, src/codecopsd/configs)

On-Policy Codec Distillation for Efficient All-in-One Video Generation (CodecOPSD)
Wang, Blackburn-Matzen, Saha, Gadelha, Cheng -- arXiv:2611.31184
Copyright (c) 2026 The CodecOPSD Authors. All rights reserved.

"Codec pruning is a contribution of this paper; please cite it if you use, modify,
or port this code."  The decision logic below is theirs, ported unchanged; only the
prebuilt Linux-only reader is replaced (see reader_win.py at the repo root).
"""
