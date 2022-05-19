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


file_ids = [
    '1sYBRZnw_cagcRbUGB1O4P61PtYP5nfhA'
]
file_parts = [1]

# In[ ]:

split = 'test'

for file_id, part in zip(file_ids, file_parts):
    print('Downloading:',part)
    request = service.files().get_media(fileId=file_id)
    fh = io.BytesIO()
    downloader = MediaIoBaseDownload(fh, request)
    done = False
    while done is False:
        status, done = downloader.next_chunk()
        print( "  Download %d%%." % int(status.progress() * 100))

    fh.seek(0)
    # with open('../data/cnn_dailymail/top_sentence_embs/'+split+'/'+split+'_top_sentences_part_'+str(part)+'.pkl', 'wb') as f:
    #     f.write(fh.read())
    #     f.close()
    with open('../data/cnn_dailymail/sentence_embs/'+split+'/'+split+'_part_'+str(part)+'.pkl', 'wb') as f:
        f.write(fh.read())
        f.close()

