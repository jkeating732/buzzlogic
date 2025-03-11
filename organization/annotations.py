import os
import sys
import json
from dotenv import load_dotenv

load_dotenv()

dataset = os.getenv("WORKING_DATASET")
project = os.getenv("WORKING_PROJECT_FILE")

if dataset is not None:
    print("Working dataset selected as " + dataset)
else:
    print("No dataset specified in .env file")
    sys.exit(1)

if project is not None:
    print("Working project selected as " + project)
else:
    print("No project file specified in .env file")
    sys.exit(1)

with open(project) as f:  # Modify project.json path as needed
    data = json.load(f)

# Count occurrences of item in labels
term = input("Class name (i.e. Mite, Queen):\n")
term_count = 0
count = 0

for task in data:
    for annotation in task.get("annotations", []):
        for result in annotation.get("result", []):
            labels = result.get("value", {}).get("rectanglelabels", [])  # Extract rectangle labels
            count += 1
            if term in labels:
                term_count += 1

            labels = result.get("value", {}).get("brushlabels", [])  # Extract brush labels
            if term in labels:
                term_count += 1

print("Total number of " + term + " annotations:", term_count)
print("Total number of annotations:", count)