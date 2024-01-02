#!/usr/bin/env python
# coding: utf-8

# In[1]:


from google.cloud import vision
import os
import os.path
import io
import pandas as pd
import numpy as np
import json
from google.protobuf.json_format import MessageToDict
from google.protobuf.json_format import MessageToJson
import proto


credential_path = "key.json goes here" 
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = credential_path


# # Face Recognition

# ## Facial Recognition

# In[31]:


def detect_face(path, max_results=4):
    """Uses the Vision API to detect faces in the given file.

    Args:
        face_file: A file-like object containing an image with faces.

    Returns:
        An array of Face objects with information about the picture.
    """
    client = vision.ImageAnnotatorClient()

    with open(path, "rb") as image_file:
        content = image_file.read()
    image = vision.Image(content=content)

    return client.face_detection(image=image, max_results=max_results).face_annotations


# In[12]:

# coded both subtask2 and subtask2b, so two paths required

path = 'add you path to the images'

sub2a = []

for dirpath, dirnames, filenames in os.walk(path):
    for filename in [f for f in filenames]:
        sub2a.append(os.path.join(dirpath, filename))


# In[13]:


sub2b = []

for dirpath, dirnames, filenames in os.walk(path):
    for filename in [f for f in filenames]:
        sub2b.append(os.path.join(dirpath, filename))


# In[26]:


images = sub2a + sub2b


# In[33]:


faces = []

for i in images:
    img_id = i.split('\\')[-1]
    output = detect_face(i)
    serializable_tags = [proto.Message.to_dict(tag) for tag in output] # regular messagetojson doesn't work so need to use this
    faces.append({'Image ID': img_id,
                'Response': serializable_tags})


# In[37]:


import json
with open('vision_face_detect.json', 'w') as fout:
    json.dump(faces, fout)


# # Web Detection

# In[40]:


web_search = []

def detect_web(path):
    """Detects web annotations given an image."""
    from google.cloud import vision

    client = vision.ImageAnnotatorClient()

    with open(path, "rb") as image_file:
        content = image_file.read()

    image = vision.Image(content=content)

    response = client.web_detection(image=image)
    annotations = response.web_detection

    return annotations


# In[62]:


web_ents = []

for i in images:
    img_id = i.split('\\')[-1]
    output = detect_web(i)
    serializable_tags = MessageToDict(output._pb) #this is the usual method
    web_ents.append({'Image ID': img_id,
                'Response': serializable_tags})


# In[63]:


with open('web_entities.json', 'w') as fout:
    json.dump(web_ents, fout)


# In[ ]:




