from GPT_SoVITS.inference_webui import change_gpt_weights, change_sovits_weights, get_tts_wav
import os
from tools.i18n.i18n import I18nAuto
import soundfile as sf

i18n = I18nAuto()

class role:
    def __init__(self, role_path):
        ref_path = os.path.join(role_path, "ref.list")
        prompt_path = os.path.join(role_path, "prompt.list")
        with open(ref_path , 'r', encoding="utf-8") as file:
            ref = file.read().split('|')
            if ref:
                self.sovits, self.gpt, self.voice, self.ref_lan, self.ref_text = ref
        with open(prompt_path, 'r', encoding="utf-8") as file:
            self.prompt = file.read()
    
    def load_model(self):
        change_gpt_weights(gpt_path=self.gpt)
        change_sovits_weights(sovits_path=self.sovits)
    
    def text_to_speech(self, output_path, language, text):
        synthesis_result = get_tts_wav(
                ref_wav_path=self.voice,
                prompt_text=self.ref_text,
                prompt_language=i18n(self.ref_lan),
                text=text,
                text_language=i18n(language),
                top_p=1,
                temperature=1,
            )
    
        result_list = list(synthesis_result)
    
        if result_list:
            last_sampling_rate, last_audio_data = result_list[-1]
            output_wav_path = os.path.join(output_path, "output.wav")
            sf.write(output_wav_path, last_audio_data, last_sampling_rate)
            print(f"Audio saved to {output_wav_path}")

            return last_audio_data, last_sampling_rate
        
  