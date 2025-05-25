from openai import OpenAI
import time
client = OpenAI(api_key="sk-e4ba477c4c624b25a1e9c55c5d75f804", base_url="https://api.deepseek.com")
import traceback

class DeepseekClient:
    def __init__(self):
        self.api_key = "sk-e4ba477c4c624b25a1e9c55c5d75f804"
        # self.session: Optional[aiohttp.ClientSession] = None
        self.message = [] # message history
        self.emotion = None
    

    def load_role(self, role):
            self.message.append({
                "role": "system",
                "content": role
            })

    
    def new_message(self, new_message, emotion='natural',language='en'):
        self.message.append({   
            "role": "user",    
            "content": f"<|{language}|><|{emotion}|><|Speech|><|withitn|>{new_message}"
        })
    
    def emotion_update(self, emotion):
        if (emotion != self.emotion):
            self.emotion= emotion
            self.message.append({"role": "system", "content": f"最新表情识别结果:\n{emotion}"})
    

    def interact_with_deepseek(self):
        try:
            print("Get answer through DeepSeek API")
            response = client.chat.completions.create(
                model="deepseek-chat",
                messages= self.message
            )

            assistant_reply = response.choices[0].message.content 
            print(f"DeepSeek Assistant: {assistant_reply}")

            # add resopnse to history 
            self.message.append({"role": "assistant", "content": assistant_reply})

            try:
                with open("assistant_reply.txt", "w", encoding="utf-8") as file:
                    file.write(assistant_reply)
                print(f"assistant_reply save to assistant_reply.txt")
            except Exception as e:
                print(f"Error when save context: {e}")
                traceback.print_exc()


        except Exception as e:
            print(f"Error at interaction with DeepSeek: {e}")
            traceback.print_exc()  
        
        return assistant_reply
