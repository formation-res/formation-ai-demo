import json

# Load the JSON data
with open('geo.json', 'r') as file:
    data = json.load(file)

# Iterate through the features and modify the structure
for feature in data['features']:
    # Extract required values
    name = feature['properties']['Name']
    _id = feature['id']
    coordinates = feature['geometry']['coordinates']

    # Construct the properties string
    properties_string = f"Name: {name}, id: {_id}, coordinates: {coordinates}"

    # Add 'properties' string under each feature
    feature['properties'] = properties_string

for feature in data['features']:
    coordinates = feature['geometry']['coordinates']
    coordinates_str = f"{coordinates[0]}, {coordinates[1]}"  # Convert coordinates to string
    feature['coordinates'] = coordinates_str


# Write the modified JSON data back to the file
with open('geo_Mod.json', 'w') as file:
    json.dump(data, file, indent=2)
    
    
