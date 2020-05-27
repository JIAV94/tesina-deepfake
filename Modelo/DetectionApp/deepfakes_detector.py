import os
os.environ["TF_CPP_MIN_LOG_LEVEL"]="2"

from pathlib import Path
import shutil
import cv2
import cvlib as cv
import glob

# PRETRAINED MODELS
from keras.models import load_model
from keras.applications.vgg16 import preprocess_input as ppi_vgg16
from keras.applications.vgg19 import preprocess_input as ppi_vgg19
from keras.preprocessing.image import ImageDataGenerator

def create_folder_structure():
    # DELETE TEST FOLDER if exists
    dirpath = Path('./', 'test')
    if dirpath.exists() and dirpath.is_dir():
        shutil.rmtree(dirpath)
    # CREATE TEST FOLDER
    Path("./test/video").mkdir(parents=True, exist_ok=True)

# Load pretrained models
model_vgg16 = load_model('vgg16-fine-tune.model')
model_vgg19 = load_model('vgg19-fine-tune.model')

Image_width, Image_height = 150, 150
batch_size = 1
width = 250
height = 250
dim = (width, height)
fakeVGG16 = 0;
fakeVGG19 = 0;
fakeAVG = 0;
realVGG16 = 0;
realVGG19 = 0;
realAVG = 0;

#EXTRACT FACE FROM VIDEO
video_path = input("\nIngresa el path de la carpeta de videos: ")
print("Este proceso puede tomar varios minutos.\n\n-------------\n")
for video in os.listdir(video_path):
    create_folder_structure()
    cap = cv2.VideoCapture(video_path+ "/" + video)
    video_name = video[:-4]
    i=0
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret == False:
            break
        faces, confidences = cv.detect_face(frame)
        for j, face in enumerate(faces):
            x1, y1 = face[0],face[1]
            x2, y2 = face[2],face[3]
            ROI = frame[y1:y2, x1:x2]
            ROI = cv2.resize(ROI, dim, interpolation = cv2.INTER_AREA)
            if ROI.any():
                cv2.imwrite('./test/video/'+video_name+'_frame_'+str(i)+'.jpg',ROI)
        i+=1
    cap.release()
    cv2.destroyAllWindows()

    # CATEGORIES = ["Real", "Fake"]

    # Predict VGG16
    test_datagen = ImageDataGenerator(rescale=1./255, preprocessing_function=ppi_vgg16)
    validate_dir = './test/'
    validation_generator = test_datagen.flow_from_directory(
        validate_dir,
        target_size=(Image_width, Image_height),
        batch_size=batch_size
    )

    nb_validation_samples = len(validation_generator.filenames)
    if(nb_validation_samples==0):
        print("No se encontraron rostros.\n")
        continue

    out = model_vgg16.predict_generator(validation_generator, nb_validation_samples)
    sum_vgg16 = 0;
    for output in out:
        if(output[0]>0.5):
            sum_vgg16 += 1
            #print(1)
            #else:
            #print(0)

    # Predict VGG19
    test_datagen = ImageDataGenerator(rescale=1./255, preprocessing_function=ppi_vgg19)
    validate_dir = './test/'
    validation_generator = test_datagen.flow_from_directory(
        validate_dir,
        target_size=(Image_width, Image_height),
        batch_size=batch_size
    )

    out = model_vgg19.predict_generator(validation_generator, nb_validation_samples)
    sum_vgg19 = 0;
    for output in out:
        if(output[0]>0.5):
            sum_vgg19 += 1
            #print(1)
        #else:
            #print(0)

    print("\n" + "Archivo analizado: " + video_name + "\n")

    if(sum_vgg16/nb_validation_samples > 0.5):
        # print("VGG16: " + str(sum_vgg16/nb_validation_samples) + " (Fake)")
        print("VGG16: (Falso)")
        fakeVGG16 += 1
    else:
        # print("VGG16: " + str(sum_vgg16/nb_validation_samples) + " (Real)")
        print("VGG16: (Real)")
        realVGG16 += 1

    if(sum_vgg19/nb_validation_samples > 0.5):
        #print("VGG19: " + str(sum_vgg19/nb_validation_samples) + " (Fake)")
        print("VGG19: (Falso)")
        fakeVGG19 += 1
    else:
        #print("VGG19: " + str(sum_vgg19/nb_validation_samples) + " (Real)")
        print("VGG19: (Real)")
        realVGG19 += 1

    '''
    if((sum_vgg16+sum_vgg19)/(nb_validation_samples*2) > 0.5):
        print("AVG: " + str((sum_vgg16+sum_vgg19)/(nb_validation_samples*2)) + " (Fake)")
        fakeAVG += 1
    else:
        print("AVG: " + str((sum_vgg16+sum_vgg19)/(nb_validation_samples*2)) + " (Real)")
        realAVG += 1
    print("-------------\n")'''

    if(sum_vgg16/nb_validation_samples > 0.5 or sum_vgg19/nb_validation_samples > 0.5):
        print("VGG16/VGG19: (Falso)\n")
        fakeAVG += 1
    else:
        print("VGG16/VGG19: (Real)\n")
        realAVG += 1
    print("-------------\n")

'''
print("---------------------------------------------")
print("VGG16\n\tReal: " + str(realVGG16) + "\n\tFalso: " + str(fakeVGG16))
print("---------------------------------------------")
print("VGG19\n\tReal: " + str(realVGG19) + "\n\tFalso: " + str(fakeVGG19))
print("---------------------------------------------")
print("VGG16/VGG19\n\tReal: " + str(realAVG) + "\n\tFalso: " + str(fakeAVG) + "\n")'''
