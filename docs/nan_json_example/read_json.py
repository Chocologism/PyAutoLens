import json
from pprint import pprint

with open('data_vector.JSON') as f:
    data = json.load(f)


print(data['autolens'])
print(data['components'][0]['plane'])
print(data['components'][0])
print(data['components'][1]['plane'])
print(data['components'][1])

