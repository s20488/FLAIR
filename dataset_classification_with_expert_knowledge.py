import os
import torch
import pandas as pd
from PIL import Image
import numpy as np
from flair import FLAIRModel

model = FLAIRModel(from_checkpoint=True)

image_folder = "/mnt/data/cfi"

categories = [
    "normal",
    "elevated blood pressure",
    "stage 1 hypertension",
    "stage 2 hypertension"
]

results = []

image_size = (224, 224)

model.eval()

for image_filename in os.listdir(image_folder):
    if image_filename.endswith(".png"):
        image_path = os.path.join(image_folder, image_filename)

        image = np.array(Image.open(image_path))

        image = Image.open(image_path).resize(image_size)
        image = np.array(image)

        with torch.no_grad():
            text_embeds_dict, text_embeds = model.compute_text_embeddings(categories, domain_knowledge=True)

            image = model.preprocess_image(image)
            img_embeds = model.vision_model(image)

            logits = model.compute_logits(img_embeds, text_embeds)
            probs = logits.softmax(dim=-1)

        logits_rounded = logits.cpu().numpy().round(3).tolist()
        probs_rounded = probs.cpu().numpy().round(3).tolist()

        results.append({
            "image": image_filename,
            "logits": logits_rounded,
            "probabilities": probs_rounded
        })

df = pd.DataFrame(results)
output_file = "./FLAIR/results_CFI_classification_with_expert_knowledge.csv"
df.to_csv(output_file, index=False)
