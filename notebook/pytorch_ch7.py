
# coding: utf-8

# # Luna
# 
# ## Packages needed
# 
# - SimpleITK
# - diskcache
# 
# ```
# !conda list --explicit > ../requirements.txt
# !conda install simpleitk -c simpleitk -y  
# !conda install -c conda-forge diskcache -y
# ```

# In[2]:


get_ipython().system('conda list --explicit > ../requirements.txt')


# In[1]:


fp_pwd = get_ipython().run_line_magic('pwd', '')
fp_ext = '/Volumes/WD/'
fp_subset = 'data/luna/subset*/{}.mhd'

# is the data in the pwd or on an external drive
data_pos = "pwd"

if data_pos is "ext":
    fp_current = fp_ext + fp_subset
else:
    fp_current = fp_pwd + fp_subset


# In[2]:


import collections
import numpy as np

IrcTuple = collections.namedtuple('IrcTuple', ['index', 'row', 'col'])
XyzTuple = collections.namedtuple('XyzTuple', ['x', 'y', 'z'])

def xyz2irc(coord_xyz, origin_xyz, vxSize_xyz, direction_tup):
    coord_cri = (np.array(coord_xyz) - np.array(origin_xyz)) / np.array(vxSize_xyz)
        
def irc2xyz(coord_irc, origin_xyz, vxSize_xyz, direction_tup):
    coord_cri = np.array(list(reversed(coord_irc)))
    coord_xyz = coord_cri * np.array(vxSize_xyz) + np.array(origin_xyz)
    return XyzTuple(*coord_cyz.tolist())


# In[3]:


import SimpleITK as sitk

class Ct(object):
    def __init__(self, series_uid):
        mhd_path = glob.glob(fp_current.format(series_uid))[0]
        
        ct_mhd = sitk.ReadImage(mhd_path)
        ct_ary = np.array(sitk.GetArrayFromImage(ct_mhd),
                          dtype=np.float32)
        
        ct_ary += 1000
        ct_ary /= 1000
        ct_ary[ct_ary < 0] = 0
        ct_ary[ct_ary > 2] = 2
        
        self.series_uid = series_uid
        self.ary = ct_ary
        self.origin_xyz = XyzTuple(*ct_mhd.GetOrigin())
        self.vxSize_xyz = XyzTuple(*ct_mhd.GetSpacing())
        self.direction_tup = tuple(int(round(x)) for x in ct_mhd.GetDirection())
        
    def getInputChunk(self, center_xyz, width_irc):
        center_irc = xyz2irc(center_xyz, self.origin_xyz, self.vxSize_xyz,
                             self.direction_tup)
        
        slice_list = []
        for axis, center_val in enumerate(center_irc):
            start_ndx = int(round(center_cal - width_irc[axis]/2))
            end_ndx = int(start_ndx + width_irc[axis])
            slice_list.append(slice(start_ndx, end_ndx))
            
            ct_chunck = self.ary[slice_list]
            
            return ct_chunk, center_irc
        


# In[4]:


class LunaDataset(Dataset):
    def __init__(self, test_stride=0, isTestSet_bool=None, series_uid=None):
        sample_list = []
        with open('data/luna/CSVFILES/candidates.csv', "r") as f:
            csv_list = list(csv.reader(f))
            
            for row in csv_list[1:]:
                row_uid = row[0]
                
                if series_uid and series_uid != row_uid:
                    continue
                    
                center_xyz = tuple([float(x) for x in row[1:4]])
                isMalignant_bool = bool(int(row[4]))
                sample_list.append((row_udi, center_xyz, isMalignant_bool))
        
        self.sample_list = sample_list
        sample_list.sort()
        if test_stride > 1:
            if isTestSet_bool:
                sample_list = sample_list[::test_stride]
            else:
                del sample_list[::test_stride]
    
    def __len__(self):
        return len(self.sample_list)
    
    def __getitem__(self, ndx):
        series_uid, center_xyz, isMalignant_bool = self.sample_list[ndx]
        ct_chunk, center_irc = getCtInputChunk(series_uid, center_xyz, (16, 16, 16))
        
        # dim=3, Undex x Row x Col
        ct_tensor = torch.from_numpy(np.array(ct_chunk,
                                              dtype=np.float32))
        
        # dim=1
        malingnant_tensor = torch.from_numpy(np.array([isMalignant_bool],
                                             dtype=np.float32))
        
        # dim=4, Channel x Index x Row x Col
        ct_tensor = ct_tensor.unsqueeze(0)
        
        # Unpacked as: input_tensor, answer_int, series_uid, center_irc
        return ct_tensor, malignant_tensor, series_uid, center_irc
    


# In[8]:


import functools
import diskcache

cache = diskcache.FanoutCache('/tmp/diskcache/fanoutcache')

@functools.lru_cache(1, typed=True)
def getCt(series_uid):
    return Ct(series_uid)

@cache.memoize(typed=True)
def getCtInputChunk(series_uid, center_xyz, width_irc):
    ct = getCt(series_uid)
    ct_chunk, center_irc = ct.getInputChunk(center_xyz, width_irc)
    return ct_chunk, center_irc


# In[3]:


# definitions not provided
get_ipython().run_line_magic('matplotlib', 'inline')
from p2ch1.vis import findMalignantSamples, showNodule
malignantSample_list = findMalignantSamples()

series_uid = malignantSample_list[7][0]
showNodule(series_uid)


# In[6]:


get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('pylab', 'inline')

import SimpleITK
import matplotlib.pyplot as plt

test_mhd = "../data/luna/subset0/1.3.6.1.4.1.14519.5.2.1.6279.6001.109002525524522225658609808059.mhd"
test_raw = "../data/luna/subset0/1.3.6.1.4.1.14519.5.2.1.6279.6001.109002525524522225658609808059.raw"

img = SimpleITK.ReadImage(test_mhd)


# In[9]:


# from: http://nbviewer.jupyter.org/urls/bitbucket.org/somada141/pyscience/raw/master/20141016_MultiModalSegmentation/Material/MultiModalSegmentation.ipynb
def sitk_show(img, title=None, margin=0.0, dpi=40):
    nda = SimpleITK.GetArrayFromImage(img)
    #spacing = img.GetSpacing()
    figsize = (1 + margin) * nda.shape[0] / dpi, (1 + margin) * nda.shape[1] / dpi
    #extent = (0, nda.shape[1]*spacing[1], nda.shape[0]*spacing[0], 0)
    extent = (0, nda.shape[1], nda.shape[0], 0)
    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax = fig.add_axes([margin, margin, 1 - 2*margin, 1 - 2*margin])

    plt.set_cmap("gray")
    ax.imshow(nda,extent=extent,interpolation=None)
    
    if title:
        plt.title(title)
    
    plt.show()


# In[14]:


idxSlice = 100

sitk_show(SimpleITK.Image(img[:, :, idxSlice]))

