from __future__ import annotations

import gc
import json
import os
import re
import traceback
import uuid
from collections.abc import Callable
from datetime import datetime, timedelta, timezone
from pathlib import Path

UTC = timezone.utc

import librosa
import numpy as np
from qwen_tts import Qwen3TTSModel
import soundfile as sf
import torch