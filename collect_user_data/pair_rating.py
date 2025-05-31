import os
import random
from PIL import Image as PILImage
from IPython.display import display

# Set path to folder containing images
image_folder = "../data/images"

# Get list of top and bottom image filenames
top_images = sorted([f for f in os.listdir(image_folder) if f.startswith('top_') and f.endswith('.jpeg')])
bottom_images = sorted([f for f in os.listdir(image_folder) if f.startswith('bottom_') and f.endswith('.jpeg')])

# Track used pairs to avoid duplicates
shown_pairs = set()

# Output file
output_file = "../data/combination_scored_v2.txt"

# Load previously logged pairs from file if it exists
if os.path.exists(output_file):
    with open(output_file, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            top_part = parts[0].split(":")[1]
            bottom_part = parts[1].split(":")[1]
            shown_pairs.add((top_part, bottom_part))


def show_pair_and_log():
    while len(shown_pairs) < len(top_images) * len(bottom_images):
        top = random.choice(top_images)
        bottom = random.choice(bottom_images)
        pair_key = (top, bottom)

        if pair_key in shown_pairs:
            continue  # Skip if already shown

        top_id = top.replace(".jpeg", "")
        bottom_id = bottom.replace(".jpeg", "")
        print(f"\nRecommendation: top:{top_id}, bottom:{bottom_id}")

        try:
            img_top = PILImage.open(os.path.join(image_folder, top))
            img_top.thumbnail((300, 300))  # Max width/height
            img_top = img_top.rotate(-90, expand=True)
            display(img_top)

            img_bottom = PILImage.open(os.path.join(image_folder, bottom))
            img_bottom.thumbnail((300, 300))
            img_bottom = img_bottom.rotate(-90, expand=True)
            display(img_bottom)

        except Exception as e:
            print(f"Failed to load/display image pair {top}, {bottom}: {e}")
            continue

        while True:
            try:
                feedback_input = input("Rate this combination (1 for good, 0 for bad, q to quit): ").strip()
                if feedback_input == "q":
                    print("Exiting by user request.")
                    return  # ends the function
                feedback = int(feedback_input)
                if feedback in [0, 1]:
                    break
                print("Invalid input. Please enter 1, 0, or q.")
            except ValueError:
                print("Please enter a numeric value or 'q' to quit.")

        with open(output_file, "a") as f:
            f.write(f"top:{top_id},bottom:{bottom_id},{feedback}\n")

        shown_pairs.add(pair_key)
        print("Saved feedback. Showing next...\n")

    print("All unique combinations have been shown.")

#Run this after running everything above on Jupyter Notebook.
show_pair_and_log()
