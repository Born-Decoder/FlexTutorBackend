from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline, SeamlessM4Tv2Model
import torch

device = "cuda:0" if torch.cuda.is_available() else "cpu"
# device = "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

model_id = "openai/whisper-large-v3"

model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
)
model.to(device)

path = "D:/Manoj/cs6460/Projects/Models/AudioToTxt/"
model.save_pretrained(path)

processor = AutoProcessor.from_pretrained("facebook/seamless-m4t-v2-large")
model2 = SeamlessM4Tv2Model.from_pretrained("facebook/seamless-m4t-v2-large")

path2 = "D:/Manoj/cs6460/Projects/Models/Translation/"
model2.save_pretrained(path2)