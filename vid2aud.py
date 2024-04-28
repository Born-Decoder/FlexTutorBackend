import moviepy.editor as mp
import os

vidfolder = 'res/Video'
audfolder = 'res/Audio'
vids = os.listdir(vidfolder)
auds = os.listdir(audfolder)
vids_to_process = [vid for vid in vids if not any(vid.split('.')[0] in aud for aud in auds)]
for vid in vids_to_process:
    clip = mp.VideoFileClip(f'{vidfolder}/{vid}')
    clip.audio.write_audiofile(f'{audfolder}/{vid.split(".")[0]}.mp3')