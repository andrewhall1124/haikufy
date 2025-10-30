
from src.haikufy.models.hugging_face_model import HuggingFaceModel
from haikufy.models.custom_model import CustomModel
from haikufy.models.local_hugging_face_model import LocalHuggingFaceModel
from dotenv import load_dotenv
import os

load_dotenv(override=True)

model_name = "meta-llama/Llama-3.2-1B-Instruct"
# model_name = "deepseek-ai/DeepSeek-V3-0324"

api_token = os.getenv("HF_TOKEN")

# model = HuggingFaceModel(model_name=model_name, api_token=api_token)
model = CustomModel(model_name=model_name, api_token=api_token)
# model = LocalHuggingFaceModel(model_name=model_name, api_token=api_token)

prompt = "What is the capital of france?"

messages = [
    {"role": "user", "content": prompt}
]

response = model.generate(
    messages=messages,
    max_tokens=50,
    temperature=.7,
    top_p=.9
)

print(response)