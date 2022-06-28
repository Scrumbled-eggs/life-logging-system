import requests
from algoliasearch.search_client import SearchClient
import json

YourApplicationID = '0L0TPDZHFM'
YourAPIKey = '1a42927a7a1ffc3661c466e3a7acda87'
your_index_name = 'milestone1'

# set up API client
client = SearchClient.create(YourApplicationID, YourAPIKey)

index = client.init_index(your_index_name)

# fetch dataset from a file
with open('./temp.json') as f:
    records = json.load(f)

# send data to Algolia
index.save_objects(records, {'autoGenerateObjectIDIfNotExist': True})

# setting searchable attributes
index.set_settings({
    'searchableAttributes': [
        'unordered(face, action)',
        'filename'
    ]
})

# setting attributes for faceting, displaying and ranking 
index.set_settings({
    'attributesForFaceting': [
        'filelength(s)',
        'fps' # fileterOnly(country) Filter-only attributes are helpful when you don’t need to offer a choice to the user.
                # searchable(country) When thousands of different values for a given facet attribute. let users search within a specific faceted attribute
    ],
    # Define business metrics for ranking and sorting
    'customRanking': [
        'asc(frame)'
    ]
})

# create replica 
index.set_settings({
  'replicas': [
    'milestone1_standard_facescore_desc',
    'milestone1_standard_actionscore_desc',
    'milestone1_standard_filename_asce'
  ]
})

replica_index = client.init_index('milestone1_standard_facescore_desc')

replica_index.set_settings({
  'ranking': [
    'desc(face_distance)',
    'typo',
    'geo',
    'words',
    'filters',
    'proximity',
    'attribute',
    'exact',
    'custom'
  ]
})

replica_index2 = client.init_index('milestone1_standard_actionscore_desc')

replica_index2.set_settings({
  'ranking': [
    'desc(action_score)',
    'typo',
    'geo',
    'words',
    'filters',
    'proximity',
    'attribute',
    'exact',
    'custom'
  ]
})

replica_index3 = client.init_index('milestone1_standard_filename_asce')

replica_index3.set_settings({
  'customRanking': [
    'asc(filename)'
  ],
  'ranking': [
    'custom',
    'typo',
    'geo',
    'words',
    'filters',
    'proximity',
    'attribute',
    'exact'
  ]
})

# set default typo tolerance mode
index.set_settings({
    'typoTolerance': 'min'
})

# remove stop words and plurals
index.set_settings({
    'queryLanguages': ['es'],
    'removeStopWords': True,
    'ignorePlurals': True
})


# save synonyms
index.save_synonym({
    'objectID': '1', # has to be string
    'type': 'oneWaySynonym',
    'input': 'walking',
    'synonyms': ['walk', 'walks', 'walked']
}, 
{'forwardToReplicas': True})

index.save_synonym({
    'objectID': '2',
    'type': 'oneWaySynonym',
    'input': 'skateboarding',
    'synonyms': ['skateboard', 'skateboards', 'riding a skateboard', 'ride a skateboard', 'riding skateboard', 'rides a skateboard', 'board']
}, 
{
    'forwardToReplicas': True
})

index.save_synonym({
    'objectID': '3',
    'type': 'oneWaySynonym',
    'input': 'makingpizza',
    'synonyms': ['cookingpizza', 'cookingpizzas', 'makingpizzas', 'bakingpizza',
    'cook a pizza, cooking a pizza, ']
}, 
{
    'forwardToReplicas': True
})

index.save_synonym({
    'objectID': '2000',
    'type': 'altCorrection1',
    'word': 'car',
    'corrections': [
        'vehicle',
        'auto'
    ]
}, {
    'forwardToReplicas': True
})

index.set_settings({
    'separatorsToIndex': '_'
})

# Filtering example: Only "Motorola" smartphones
#results = index.search('smartphone', {
#    'filters': 'brand:Motorola'
#})
