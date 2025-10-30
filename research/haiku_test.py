from rich import print

from src.haikufy.converter import HaikuConverter

hc = HaikuConverter(model_wrapper='hf-custom')

prompt = "Do you want to run today?"

haiku = hc.generate_haiku(text=prompt)

print(haiku)




