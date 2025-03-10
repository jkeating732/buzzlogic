import json

with open("project.json") as f:
    data = json.load(f)

# Count occurrences of item in rectanglelabels
term = input("Class name (i.e. Mite, Queen):\n")
term_count = 0
count = 0

for task in data:
    for annotation in task.get("annotations", []):
        for result in annotation.get("result", []):
            labels = result.get("value", {}).get("rectanglelabels", [])  # Extract labels
            count += 1
            if term in labels:
                term_count += 1

print("Total number of " + term + " annotations:", term_count)
print("Total number of annotations:", count)

