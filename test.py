import os
import logging
import re
from datasets import load_from_disk, load_metric, Audio
import json
import numpy as np
import torch
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
from transformers import Wav2Vec2CTCTokenizer, Wav2Vec2FeatureExtractor, Wav2Vec2Processor, Wav2Vec2ForCTC, TrainingArguments, Trainer

print(torch.cuda.is_available())