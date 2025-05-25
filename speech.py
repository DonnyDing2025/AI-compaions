from funasr import AutoModel
import whisper

# import subprocess

# subprocess.run(r'$env:Path += ";D:\\work\\ECE479\\lab3\\ffmpeg\\bin"', shell=True)# use your local directory
class AsrModel: # build model for interpretation
    def __init__(self):
        self.zh_model = AutoModel(
            model="STT_pack/paraformer-zh",
            vad_model="STT_pack/fsmn-vad",
            punc_model="STT_pack/ct-punc",
            device="cuda",
            disable_update=True
        )
        self.en_model = AutoModel(
            model="STT_pack/paraformer-en",
            vad_model="STT_pack/fsmn-vad",
            punc_model="STT_pack/ct-punc",
            device="cuda",
            disable_update=True
        )
        self.emotion_model = AutoModel(
            model="STT_pack/emotion2vec+base",
            granularity="utterance",
            extract_embedding=False,
            device="cuda",
            disable_update=True
        )        
        self.check_model = whisper.load_model("base")
    
    def language_identification(self, audio_chunk):
        samp_audio = whisper.pad_or_trim(audio_chunk)

        # make log-Mel spectrogram and move to the same device as the model
        mel = whisper.log_mel_spectrogram(samp_audio, n_mels=self.check_model.dims.n_mels).to("cuda")
        # detect the spoken language
        _, probs = self.check_model.detect_language(mel)
        return max(probs, key=probs.get)
    
    def emotion_sense(self, audio_chunk):
        result = self.emotion_model.generate(audio_chunk)[0]
        labels = result["labels"]
        scores = result["scores"]
        return labels[scores.index(max(scores))].split('/')
    
    def trabscribe(self, audio_chunk, current_lang):
        if current_lang == "zh":
            return self.zh_model.generate(audio_chunk)[0]['text']
        else:
            return self.en_model.generate(audio_chunk)[0]['text']

# import time
# from pathlib import Path

# # 假设您的 AsrModel 类定义在 asr_model.py 文件中

# def test_asr_model(audio_path: str):
#     # 初始化模型（首次运行会下载模型，可能需要较长时间）
#     print("Loading models...")
#     start_time = time.time()
#     asr_model = AsrModel()
#     print(f"Models loaded in {time.time() - start_time:.2f}s\n")

#     # 检查文件是否存在
#     if not Path(audio_path).exists():
#         raise FileNotFoundError(f"Audio file {audio_path} not found")

#     try:
#         # 语言识别测试
#         print("Testing language identification...")
#         audio = whisper.load_audio(audio_path)
#         lang = asr_model.language_identification(audio)
#         print(f"Detected language: {lang}\n")

#         # 情感分析测试
#         print("Testing emotion recognition...")
#         emotion = asr_model.emotion_sense(audio_path)
#         print(f"Detected emotion: {emotion}\n")

#         # 语音识别测试
#         print("Testing speech recognition...")
#         transcript = asr_model.trabscribe(audio_path, current_lang=lang)
#         print(f"Transcript: {transcript}")

#     except Exception as e:
#         print(f"Error during processing: {str(e)}")
#         raise

# if __name__ == "__main__":

#     audio_file = "./asr_example.wav"
#     print(f"Testing ASR system with audio file: {audio_file}")
#     test_asr_model(audio_file)

        

        
    


