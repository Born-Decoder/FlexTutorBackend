## About

The app is called FlexTutor that uses Text-to-speech, Speech-to-text, Multi-language translation models, automatic video summarization techniques and large language models to augment users by providing them effective means of learning and accessing educational content in their own comfortable way they prefer. This would be an Android application that can process videos, audio and video content to help the users and assist them in their learning process.

## This directory contains Flask setup for app backend

### Requirements

- python modules - `flask, google, transformers, torch, moviepy, ffmpeg`
- install ffmpeg in local https://www.wikihow.com/Install-FFmpeg-on-Windows
- Download sample data video(s) (ideally the intermediate milestone videos) that should match with resids.json entries
- Milestone 1 video - https://drive.google.com/file/d/10KM8Wxx3txqbSRIyFMxftQTHtR6r0JWR/view?usp=sharing
- Milestone 2 video - https://drive.google.com/file/d/1tjPET3gUS7Ws70PenHQh0b-G-46Ed6bO/view?usp=sharing
- Note - If other videos are used, then resids.json metadata should be adjusted
- Videos should be placed under `res/Video/`
- Add gemini credentials in `credentials.json` in the folloeing format `{
    "gemini-key": "<yourkey>"
}`
- The IDs in `resids.json` should match the youtube video that is being referred, if a different data is used
### Steps to run

- Adjust the model directories where you want to save your models in `savemodels.py`
- Run `savemodels.py` and confirm if the model files are accessible in the necessary target directories
- Run `vid2aud.py` to generate audio files from videos using `ffmpeg`. The audio files should be present in `res/Audio/`
- run `Aud2txt.py` to generate text and transcriptions for your audio content into `res/Text/`
- Check your host IP and model directories in `app.py` and run to start hosting flask app
