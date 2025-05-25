import subprocess
subprocess.run("$env:Path += ';D:\\work\\ECE479\\lab3\\ffmpeg\\bin'", shell=True)# use your local directory
from speech import AsrModel
from audio import voice_sense
from deepseek_interaction import DeepseekClient
from role import role
import pyaudio
import keyboard
import sounddevice as sd
import numpy as np
from Emotion import EmotionModel
role_path = "./role/Mari"
output_path= "./output"

def image_caputre():
     pass


def main():
    try:
        asr = AsrModel()
        # emotion_model = EmotionModel()
        character = role(role_path)
        character.load_model()
        client = DeepseekClient()
        client.load_role(character.prompt)
        while True:  
            #录音和语音识别
            print("开始录音")
            
            audio = pyaudio.PyAudio()
            stream = audio.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=16000,
                input=True,
                frames_per_buffer=320
            )
            audio = voice_sense(stream)
            audio.terminate()
            emotion = asr.emotion_sense
            lang = asr.language_identification()
            message = asr.trabscribe(audio, lang)  
            print(f"text result: {message}")
            client.emotion_update(emotion)
            # message = input("enter conversation message ")
            # lang = "英文"
            # emotion = 'happy'
            client.new_message(message,language=lang[1],emotion=emotion)     
            reply = client.interact_with_deepseek()

            # audio transciption
            audio, sample_rate = character.text_to_speech(output_path, lang[0], reply)
            output_device = 11 # choose VB Input on own machine
            sd.play(audio, sample_rate, device=output_device)
            sd.wait()

            user_continue = input("conitnue? (y/n): ")  
            if user_continue.lower() != "y":   
                break
        print("End.")  

    except Exception as e:
        print(f"程序运行中出现错误: {e}")

if __name__ == "__main__":  
    main()  