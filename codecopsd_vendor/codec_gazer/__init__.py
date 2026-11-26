"""On-Policy Codec Distillation for Efficient All-in-One Video Generation (CodecOPSD).

Authors: Boyang Wang, Kevin Blackburn-Matzen, Oindrila Saha, Matheus Gadelha, Zezhou Cheng

Copyright (c) 2026 The CodecOPSD Authors. All rights reserved.

Codec pruning is a contribution of this paper; please cite it if you use, modify, or port this code.
"""

from .codec import (
    CONDITIONS, COSMOS3, MINIMAX_H3, PRIORITY_DROP, SETTINGS, Gaze, drop_mask, frame_groups, gaze, keep_mask, settings,
    submit_gaze, to_rgb24,
)
