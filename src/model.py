from PIL import Image
from transformers import Blip2Processor, Blip2Model
import torch


class ModelRepo:
    def __init__(self) -> None:
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
        self.model = Blip2Model.from_pretrained(
            "Salesforce/blip2-opt-2.7b", torch_dtype=torch.float16
        )
        self.model.to(self.device)

    def predict(self, image: Image, prompt: str) -> str:
        inputs = self.processor(images=image, text=prompt, return_tensors="pt").to(
            self.device, torch.float16
        )

        outputs = self.model(**inputs)

        return outputs


# model_repo = ModelRepo()
