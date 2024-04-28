import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import os

device = "cuda:0" if torch.cuda.is_available() else "cpu"
# device = "cpu"
model = AutoModelForSpeechSeq2Seq.from_pretrained("D:\Manoj\cs6460\Projects\Models\AudioToTxt")
model_id = "openai/whisper-large-v3"
processor = AutoProcessor.from_pretrained(model_id)
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
torch_dtype = torch.float32
model.to(device)
pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    max_new_tokens=128,
    chunk_length_s=30,
    batch_size=16,
    return_timestamps=True,
    torch_dtype=torch_dtype,
    device=device,
)

audfolder = 'res/Audio'
txtfolder = 'res/Text'
auds = os.listdir(audfolder)
txts = os.listdir(txtfolder)
auds_to_process = [aud for aud in auds if not any(aud.split('.')[0] in txt for txt in txts)]
for aud in auds_to_process:
    print(f"Running on {audfolder}/{aud}")
    audiopipe = pipe(f"{audfolder}/{aud}", generate_kwargs={"language": "english"})
    print(f'Saving to {txtfolder}/{aud.split(".")[0]}.txt')
    with open(f'{txtfolder}/{aud.split(".")[0]}.txt', "w") as text_file:
        text_file.write(audiopipe['text'])