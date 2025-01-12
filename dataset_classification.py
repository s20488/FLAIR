import os

import pandas as pd
from PIL import Image
import numpy as np
from flair import FLAIRModel

model = FLAIRModel(from_checkpoint=True)

image_folder = "/mnt/data/cfi"

text = ["normal", "elevated blood pressure", "stage 1 hypertension", "stage 2 hypertension"]

results = []

for image_filename in os.listdir(image_folder):
    if image_filename.endswith(".png"):
        image_path = os.path.join(image_folder, image_filename)

        image = np.array(Image.open(image_path))

        probs, logits = model(image, text)

        logits_rounded = logits.round(3).tolist()
        probs_rounded = probs.round(3).tolist()

        results.append({
            "image": image_filename,
            "logits": logits_rounded,
            "probabilities": probs_rounded
        })

df = pd.DataFrame(results)

output_file = "./FLAIR/results_CFI_classification.csv"
df.to_csv(output_file, index=False)
