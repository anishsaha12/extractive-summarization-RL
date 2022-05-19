#!/usr/bin/env python
# coding: utf-8

# In[1]:


from Google import Create_Service
from googleapiclient.http import MediaFileUpload
import os
import io


# In[2]:


CSF = 'client_secret.json'
API_NAME = 'drive'
API_VERSION = 'v3'
SCOPES = ['https://www.googleapis.com/auth/drive']


# In[3]:


service = Create_Service(CSF, API_NAME, API_VERSION, SCOPES)


# In[ ]:


file_paths = [
    '../data/cnn_dailymail/dataset_dict.json'
]
file_names = [
    'dataset_dict.json'
]

# In[ ]:
        
folder_id = '1DM6Wf8gd4_7yv36uw8aKkkXVmQC9OYEL'

for file_path, file_name in zip(file_paths, file_names):
    file_metadata = {
        'name': file_name,
        'parents': [folder_id]
    }
    media = MediaFileUpload(file_path)
    file = service.files().create(body=file_metadata,
                                        media_body=media,
                                        fields='id').execute()
    print('File ID: %s' % file.get('id'))