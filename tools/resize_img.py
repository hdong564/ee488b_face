import os,sys
from PIL import Image


size = 256, 256 
path = "/mnt/home/20180523/ee488b_face/data/vggface2/train"
modified_path = "/mnt/home/20180523/ee488b_face/data/resized_vggface2"
os.chdir(path)



def resize_and_crop(img_path, modified_path, size, crop_type='middle'):
    
    directories = os.listdir(img_path)
    for direct in directories: 
        os.chdir(direct)
        os.mkdir(modified_path + '/'+ direct)
        print("current dir: " ,os.getcwd())
        files = os.listdir()

        for file in files:
            name = str(file)
            img = Image.open(file)
            img_ratio = img.size[0] / float(img.size[1])
            ratio = size[0] / float(size[1])
            
            if ratio > img_ratio:
                img = img.resize((size[0], int(round(size[0] * img.size[1] / img.size[0]))),
                    Image.ANTIALIAS)     
                if crop_type == 'top':
                    box = (0, 0, img.size[0], size[1])
                elif crop_type == 'middle':
                    box = (0, int(round((img.size[1] - size[1]) / 2)), img.size[0],
                        int(round((img.size[1] + size[1]) / 2)))
                elif crop_type == 'bottom':
                    box = (0, img.size[1] - size[1], img.size[0], img.size[1])
                else :
                    raise ValueError('ERROR: invalid value for crop_type')
                img = img.crop(box)
                
            elif ratio < img_ratio:
                img = img.resize((int(round(size[1] * img.size[0] / img.size[1])), size[1]),
                    Image.ANTIALIAS)
                if crop_type == 'top':
                    box = (0, 0, size[0], img.size[1])
                elif crop_type == 'middle':
                    box = (int(round((img.size[0] - size[0]) / 2)), 0,
                        int(round((img.size[0] + size[0]) / 2)), img.size[1])
                elif crop_type == 'bottom':
                    box = (img.size[0] - size[0], 0, img.size[0], img.size[1])
                else :
                    raise ValueError('ERROR: invalid value for crop_type')
                img = img.crop(box)
                
            else :
                img = img.resize((size[0], size[1]), Image.ANTIALIAS)
                
            os.chdir(modified_path +'/'+  direct)
            img.save(name)
            os.chdir(img_path+'/'+direct) # goback to train dataset directory
        print("@ searching {} files done!".format(direct))    
        os.chdir(path)

resize_and_crop(path, modified_path, size)