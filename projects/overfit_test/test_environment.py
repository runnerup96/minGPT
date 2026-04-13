import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import torch
import numpy as np
import transformers
import requests
from mingpt.utils import CfgNode as CN
from mingpt.model import GPT
from mingpt.enc_dec_model import EncoderDecoderGPT

print("All imports OK")
print(f"  torch       {torch.__version__}")
print(f"  numpy       {np.__version__}")
print(f"  transformers {transformers.__version__}")
