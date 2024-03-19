import json


# USE: Example of how to modify a generic JSON file cUrled from the internet
# takes file Formation1.json, to Formation1_Mod.json

# Load the JSON data
with open('Formation1.json', 'r') as file:
    data = json.load(file)
    
    
for hits in data["hits"]:
    tags = [
        tag for tag in hits["hit"]["tags"]
        if not tag.startswith(("IconCategory:", "Color:", "Shape:", "ConnectedTo:",
                            "GroupId:", "ConnectedObjectType:", "BuildingId:", "Outdooe:", "OwnerId:", "Creator:"))
    ]
    
    page_content = "Name: " + "\n" + hits["hit"]["title"] + ": " + ", ".join(tags)
    hits["hit"]["page_content"] = page_content


# Write the modified JSON data back to the file
with open('Formation1_Mod.json', 'w') as file:
    json.dump(data, file, indent=2)
    
    





    
