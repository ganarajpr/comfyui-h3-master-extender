"""On-Policy Codec Distillation for Efficient All-in-One Video Generation (CodecOPSD).

Authors: Boyang Wang, Kevin Blackburn-Matzen, Oindrila Saha, Matheus Gadelha, Zezhou Cheng

Copyright (c) 2026 The CodecOPSD Authors. All rights reserved.

Settings of the released code, one commented YAML file per topic (`load("codec_gazer")`)."""

from functools import cache
from importlib.resources import files

import yaml


@cache
def load(name):
    """The settings in `src/codecopsd/configs/<name>.yaml` as a dict."""
    return yaml.safe_load(files(__package__).joinpath(f"{name}.yaml").read_text())
