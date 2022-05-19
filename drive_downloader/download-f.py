#!/usr/bin/env python
# coding: utf-8

# In[1]:


from Google import Create_Service
from googleapiclient.http import MediaIoBaseDownload
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


file_ids = ['1KhqvR2nS2CSNshMebc-Z2-mAXVhnHxZ1']
file_parts = ['../data/cnn_dailymail/sentences/test/test_rewards.gz']

# In[ ]:

split = 'test'

for file_id, name in zip(file_ids, file_parts):
    print('Downloading:',name)
    request = service.files().get_media(fileId=file_id)
    fh = io.BytesIO()
    downloader = MediaIoBaseDownload(fh, request)
    done = False
    while done is False:
        status, done = downloader.next_chunk()
        print( "  Download %d%%." % int(status.progress() * 100))

    fh.seek(0)
    with open(name, 'wb') as f:
        f.write(fh.read())
        f.close()

