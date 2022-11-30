from PIL import Image

path = '/mnt/home/20180523/ee488b_face/data/kor_VGGface2/train/id10051/'

for i in range(1,333):
    jpgname = 'B0000000{}.jpg'.format(i)
    # print(path + jpgname)
    fname = path + jpgname
    img = Image.open(fname)
    print(img)