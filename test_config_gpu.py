#!/usr/bin/env python3
"""Test config GPU setting."""

import os
from bu_processor.bu_processor.core.config import get_config

print(f"BU_USE_GPU environment variable: {os.environ.get('BU_USE_GPU', 'not set')}")
cfg = get_config()
print(f"Config use_gpu value: {cfg.ml_model.use_gpu}")
print(f"Config use_gpu type: {type(cfg.ml_model.use_gpu)}")
