import torch
from qwen_tts import Qwen3TTSModel

m = Qwen3TTSModel.from_pretrained(
    r"C:\Users\jsouz\Projects\nova-generator\models\qwen3-tts\Qwen3-TTS-12Hz-1.7B-Base",
    device_map="cpu",
    dtype=torch.bfloat16,
    low_cpu_mem_usage=False,
)
print("OK")