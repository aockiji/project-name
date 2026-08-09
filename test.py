"""
for now what this does is it takes an image and runs the run_all method from segmentation.py which
the run_all method runs the four segmentation methods we're usinf and then it saves the resuls as the name
of the segmethod used plus .png and puts them all in a folder called "test-[num]-res" where num is an autogenend number
will be changing the tset images alnog the way
also make a ver where we run on several images in a specific path at once


"""
import numpy as np
from PIL import Image
from segmentation import run_all

img = Image.open("wano.jpg").convert("RGB")
img_array = np.array(img)

print(f"Loaded image: {img_array.shape}")
print("Running all 4 algos...")

results = run_all(img_array)

from pathlib import Path

base_dir = Path(".")
test_number = 1

while (base_dir / f"test-{test_number}-res").exists():
    test_number += 1

output_dir = base_dir / f"test-{test_number}-res"
output_dir.mkdir()

for name, data in results.items():
    print(f"{name}: runtime={data['runtime']}s, output shape={data['image'].shape}")

    output_image = Image.fromarray(data["image"])
    filename = f"{name.replace(' ', '_')}.png"
    output_image.save(output_dir / filename)

print(f"\nDone! Results saved in '{output_dir}/'.")