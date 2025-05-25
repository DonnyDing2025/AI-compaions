import pyaudio
import webrtcvad
import numpy as np
import keyboard

# FORMAT = pyaudio.paInt16
# CHANNELS = 1
# RATE = 16000  
# FRAME_DURATION = 20  
# CHUNK = int(RATE * FRAME_DURATION / 1000)  
# SILENCE_TIMEOUT = 1
# THRESHOLD_SILENCE_FRAMES = int(SILENCE_TIMEOUT / (FRAME_DURATION / 1000))

def voice_sense(stream, rate=16000, chunk=320, silent_frames=50):
    vad = webrtcvad.Vad()
    vad.set_mode(3)  # 检测灵敏度（0~3，3最严格）
    # in external port
    # audio = pyaudio.PyAudio()
    # stream = audio.open(
    #     format=FORMAT,
    #     channels=CHANNELS,
    #     rate=RATE,
    #     input=True,
    #     frames_per_buffer=CHUNK
    # )

    frames = []
    silent_frame = 0
    is_speaking = False
    pre_buffer = []
    pre_buffer_max = 5  
    try:
        while True:
            if keyboard.is_pressed('space'):  # 录音过程中检查空格键
                print("end recoding...")
                break
            data = stream.read(chunk)
            pre_buffer.append(data)
            if len(pre_buffer) > pre_buffer_max:
                pre_buffer.pop(0)
            is_speech = vad.is_speech(data, rate)

            if is_speech:
                silent_frame = 0
                if not is_speaking:
                    frames.extend(pre_buffer) 
                    is_speaking = True
            else:
                if is_speaking:
                    silent_frame += 1
                    if silent_frame > silent_frames:
                        print("end recoding...")
                        break

            frames.append(data)
    except KeyboardInterrupt:
        pass

    stream.stop_stream()
    stream.close()
    # audio.terminate()
    voice = np.concatenate(data, axis=0)
    return voice


