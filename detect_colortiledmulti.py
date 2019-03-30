# import the necessary packages
from PIL import Image
import numpy as np
import argparse
import cv2
import os
import multiprocessing
import time
def long_slice(image_path,outdir,filename, sliceHeight, sliceWidth):
    img = Image.open(image_path) # Load image
    imageWidth, imageHeight = img.size # Get image dimensions
    left = 0 # Set the left-most edge
    upper = 0 # Set the top-most edge
    #imagelist = []
    count=0
    print('started tiling')
    while (left < imageWidth):
        while (upper < imageHeight):
            # If the bottom and right of the cropping box overruns the image.
            if (upper + sliceHeight > imageHeight and \
                left + sliceWidth > imageWidth):
                bbox = (left, upper, imageWidth, imageHeight)
            # If the right of the cropping box overruns the image
            elif (left + sliceWidth > imageWidth):
                bbox = (left, upper, imageWidth, upper + sliceHeight)
            # If the bottom of the cropping box overruns the image
            elif (upper + sliceHeight > imageHeight):
                bbox = (left, upper, left + sliceWidth, imageHeight)
            # If the entire cropping box is inside the image,
            # proceed normally.
            else:
                bbox = (left, upper, left + sliceWidth, upper + sliceHeight)
            working_slice = img.crop(bbox) # Crop image based on created bounds
            #imagelist.append(working_slice)
            # Save your new cropped image.
            working_slice.save(os.path.join(outdir, filename+str(count)+'.jpg'))
            upper += sliceHeight # Increment the horizontal position
            count+=1
        left += sliceWidth # Increment the vertical position

        upper = 0
def long_slicetest(image_path,outdir,filename, sliceHeight, sliceWidth):
    img = Image.open(image_path) # Load image
    imageWidth, imageHeight = img.size # Get image dimensions
    left = 0 # Set the left-most edge
    upper = 0 # Set the top-most edge
    #imagelist = []
    count=0
    print('started tiling')
    overlap=500
    while (left+overlap < imageWidth):
        while (upper+overlap < imageHeight):
            # If the bottom and right of the cropping box overruns the image.
            if (upper + sliceHeight+overlap > imageHeight and
                left + sliceWidth+overlap > imageWidth):
                bbox = (left+overlap, upper+overlap, imageWidth, imageHeight)
            # If the right of the cropping box overruns the image
            elif (left + sliceWidth+overlap > imageWidth):
                bbox = (left+overlap, upper+overlap, imageWidth, upper + sliceHeight+overlap)
            # If the bottom of the cropping box overruns the image
            elif (upper + sliceHeight+overlap > imageHeight):
                bbox = (left+overlap, upper+overlap, left + sliceWidth+overlap, imageHeight)
            # If the entire cropping box is inside the image,
            # proceed normally.
            else:
                bbox = (left+overlap, upper+overlap, left + sliceWidth+overlap, upper + sliceHeight+overlap)
            working_slice = img.crop(bbox) # Crop image based on created bounds
            #imagelist.append(working_slice)
            # Save your new cropped image.
            working_slice.save(os.path.join(outdir, filename+str(count)+'.jpg'))
            upper += sliceHeight # Increment the horizontal position
            count+=1
        left += sliceWidth # Increment the vertical position

        upper = 0
tilecount1=0
tilecount2=1
tilecount3=2
tilecount4=3

def process1(directory,tilecount):

    for file in os.listdir(directory):

        if tilecount == 3:
            #ap = argparse.ArgumentParser()
            #ap.add_argument("-i", "--image", help=file)
            file=str(file)
            file=file.replace("b", "")
            filename=file.replace("'", "")
            file = rootdir + "/" + filename
            filename2 = filename.replace(".JPG", "")

            #file = 'C:/Users/twburton/Desktop/trainingpic4/colortest/42.png'
            #image = cv2.imread(file)
            #long_slice(file,outdirectory3, 2896, 4344)# divided by 4
            #long_slice(file,outdirectory3,filename2, 1448, 2172)# divided by

            #long_slice(file,outdirectory3,filename2, 1872, 2808)#5616 x 3744 divided by 2
            #long_slice(file,outdirectory3,filename2, 936, 1404)#5616 x 3744 divided by 4
            long_slice(file,outdirectory3,filename2, 468, 702)#5616 x 3744 divided by 8
            #long_slice(file,outdirectory3,filename2, 234, 351)#5616 x 3744 divided by 16
            tilecount = 0
        else:
            tilecount += 1

#import matplotlib.pyplot as plt
# construct the argument parse and parse the arguments
#ap = argparse.ArgumentParser()
#ap.add_argument("-i", "--image", help = "C:/Users/twburton/Desktop/trainingpic3/b/31.png")
#args = vars(ap.parse_args())

# rootdir = 'C:/Users/twburton/Desktop/output/test2'
# outdirectory = 'C:/Users/twburton/Desktop/output/birds/'
# outdirectory2 = 'C:/Users/twburton/Desktop/output/nobirds/'
# outdirectory3 = 'C:/Users/twburton/Desktop/output/tiles/'
rootdir = 'F:/trainingpic4/testimages'
outdirectory = 'F:/trainingpic4/output/'
outdirectory2 = 'F:/trainingpic4/outputempty/'
outdirectory3 = 'F:/trainingpic4/tiles/'
# load the image
#image = cv2.imread(args["image"])

# define the list of boundaries
boundaries1 = [#G,B,R
	([1, 1, 1], [48, 48, 48]),#black
	([210, 210, 210], [255, 255, 255]),#white
([35, 1, 65], [90, 55, 180]),#brown
([65, 10, 140], [130, 90, 185]),#brown2
([76, 76, 86], [79, 79, 89])#brown3
]

boundaries2 = [
	([1, 1, 1], [35, 35, 35]),#black
	([200, 200, 200], [255, 255, 255]),#white
([35, 1, 65], [90, 55, 180]),#brown
([65, 10, 140], [130, 90, 185]),#brown2
([76, 79, 86], [76, 79, 89])#brown3

]
# loop over the boundaries
directory = os.fsencode(rootdir)
dircount1=0
dircount2=1
dircount3=2
dircount4=3

def process2(directory,outdirectory,outdirectory2,outdirectory3,rootdir,boundaries1,boundaries2,dircount):

    for file in os.listdir(directory):
        breaker = False

        if dircount == 3:
            #ap = argparse.ArgumentParser()
            #ap.add_argument("-i", "--image", help=file)
            file=str(file)
            file=file.replace("b", "")
            filename=file.replace("'", "")
            file = rootdir + "/" + filename
            print(file)
            filename2 = filename.replace(".JPG", "")
            #print(filename+"aaaaa")
            #print("aaa"+str(dircount))

            #file = 'C:/Users/twburton/Desktop/trainingpic4/colortest/42.png'
            image = cv2.imread(file)
            #accumMask = np.zeros(image.shape[:2], dtype="uint8")

            count = 0

            #long_slice(file,outdirectory3, 2896, 4344)# divided by 4
            #long_slice(file,outdirectory3, 1448, 2172)# divided by 16
            for i in range (0,64):#change to number of tiles

                file2 = outdirectory3 +filename2+str(i)+".jpg"
                imagetile = cv2.imread(file2)
                colorcount = 0
                print(i)
                if (i==6) or (i==7) or (i==22)or (i==23)or (i==38)or (i==39)or (i==54)or (i==55)or (i==15)or (i==31)or (i==47)or (i==63):# for 64 tiles
                    boundaries=boundaries2#change to add second color list
                else:
                    boundaries=boundaries1
                for (lower, upper) in boundaries:

                    # create NumPy arrays from the boundaries
                    lower = np.array(lower, dtype="uint8")
                    upper = np.array(upper, dtype="uint8")

                    # find the colors within the specified boundaries and apply
                    # the mask
                    mask = cv2.inRange(imagetile, lower, upper)
                    #output = cv2.bitwise_and(imagetile, imagetile, mask=mask)
                    # show the images
                    # merge the mask into the accumulated masks
                    #accumMask = cv2.bitwise_or(accumMask, mask)

                    #accumMask = cv2.bitwise_not(accumMask)
                    unmasked = cv2.countNonZero(mask)
                    colorvar = ""
                    if colorcount == 0:
                        colorvar = "black "
                    if colorcount == 1:
                        colorvar = "white "
                    if colorcount == 2:
                        colorvar = "brown "
                    if colorcount == 3:
                        colorvar = "brown2 "
                    if colorcount == 4:
                        colorvar = "brown3 ---------------"
                    if unmasked:
                        print(colorvar+str(unmasked))
                        if (unmasked>10):#more pixels required for certain color?

                            cv2.imwrite(outdirectory+filename, image)
                            breaker=True
                            break
                        else:
                            count+=1
                    else:
                        print("none "+colorvar+str(unmasked))
                        count+=1

                    colorcount += 1
                if breaker:
                    break
            #if count==80:#change for more colors
            if breaker==False:
                cv2.imwrite(outdirectory2 + filename, image)
            dircount = 0
        else:
            dircount += 1


if __name__ =='__main__':
    p1 = multiprocessing.Process(target=process1, args=(directory,tilecount1))
    p11 = multiprocessing.Process(target=process1, args=(directory,tilecount2))
    p12 = multiprocessing.Process(target=process1, args=(directory,tilecount3))
    p13 = multiprocessing.Process(target=process1, args=(directory,tilecount4))
    p1.start()
    p11.start()
    p12.start()
    p13.start()
    p1.join()
    p11.join()
    p12.join()
    p13.join()

    p2 = multiprocessing.Process(target=process2, args=(directory,outdirectory,outdirectory2,outdirectory3,rootdir,boundaries1,boundaries2,dircount1))
    p3 = multiprocessing.Process(target=process2,args=(directory, outdirectory, outdirectory2, outdirectory3, rootdir, boundaries1,boundaries2,dircount2))
    p4 = multiprocessing.Process(target=process2,args=(directory, outdirectory, outdirectory2, outdirectory3, rootdir, boundaries1,boundaries2,dircount3))
    p5 = multiprocessing.Process(target=process2,args=(directory, outdirectory, outdirectory2, outdirectory3, rootdir, boundaries1,boundaries2,dircount4))
    p2.start()
    p3.start()
    p4.start()
    p5.start()

