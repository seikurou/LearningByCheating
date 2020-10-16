import lmdb
import cv2
from tqdm import tqdm
import numpy as np
import PIL
from color import colorize
import os
import glob

if not os.path.exists('imgs'):
    os.mkdir('imgs')

#road lane traffic vehicle pedestrian

folders = sorted([k for k in glob.glob('*') if os.path.exists(k +'/data.mdb')])

for lmdb_dir in tqdm(folders, desc='Folder'):
    if not os.path.exists('imgs/' + lmdb_dir):
        os.mkdir('imgs/' + lmdb_dir)
    lmdb_file = lmdb.open(lmdb_dir,
            max_readers=1,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False
    )
    txn = lmdb_file.begin(write=False)
    N = int(txn.get('len'.encode()))
    for i in tqdm(range(N), desc='Frame'):
        img = cv2.imdecode(np.frombuffer(txn.get(('rgb_%04d'%i).encode()), np.uint8), cv2.IMREAD_UNCHANGED)
        img_left = cv2.imdecode(np.frombuffer(txn.get(('rgb_left_%04d'%i).encode()), np.uint8), cv2.IMREAD_UNCHANGED)
        img_right = cv2.imdecode(np.frombuffer(txn.get(('rgb_right_%04d'%i).encode()), np.uint8), cv2.IMREAD_UNCHANGED)
        # print(img.shape)
        depth = cv2.imdecode(np.frombuffer(txn.get(('depth_%04d'%i).encode()), np.uint8), cv2.IMREAD_UNCHANGED)
        semantic_arr = cv2.imdecode(np.frombuffer(txn.get(('semantic_%04d'%i).encode()), np.uint8), cv2.IMREAD_UNCHANGED)
        bv = np.frombuffer(txn.get(('birdview_%04d'%i).encode()), np.uint8).reshape(320, 320, 7) 


        # semantic = np.zeros((512, 1250, 3))
        # for row in range(512):
        #     for col in range(1250):
        #         semantic[row][col] = segmentation[semantic_arr[row][col]]
        semantic = colorize(semantic_arr)
        
        # img =cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.copyMakeBorder(img, 0, 512, 1280, 1280, cv2.BORDER_CONSTANT)
        

        img[512:512+320,1280*2:1280*2+320,0] = bv[...,0]
        img[512:512+320,1280*2:1280*2+320,1] = bv[...,1]
        img[512:512+320,1280*2:1280*2+320,2] = bv[...,5]
        img[:512, :1280] = img_left
        img[:512, 1280*2:1280*3] = img_right
        img[512:512*2,:1280] = depth[...,None] // 256
        img[512:512*2,1280:1280*2] = semantic[...,::-1]
        cv2.imwrite('imgs/' + lmdb_dir + '/%04d.png'%i, img)
        
        
