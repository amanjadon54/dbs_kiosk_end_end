# This is a demo of running face recognition on a Raspberry Pi.
# This program will print out the names of anyone it recognizes to the console.

# To run this, you need a Raspberry Pi 2 (or greater) with face_recognition and
# the picamera[array] module installed.
# You can follow this installation instructions to get your RPi set up:
# https://gist.github.com/ageitgey/1ac8dbe8572f3f533df6269dab35df65

import face_recognition
import picamera
import numpy as np

import subprocess

import time


#the below for QR
import zbarlight
import os
import sys
from PIL import Image


#below for gesture detection
import cv2
from keras.models import load_model
import picamera.array 

# Get a reference to the Raspberry Pi camera.
# If this fails, make sure you have a camera connected to the RPi and that you
# enabled your camera in raspi-config and rebooted first.

# Load a sample picture and learn how to recognize it.
print("Loading known face image(s)")
ashhad_image = face_recognition.load_image_file("ashhad_sml.jpg")
ashhad_face_encoding = face_recognition.face_encodings(ashhad_image)[0]

# Initialize some variables
face_locations = []
face_encodings = []

##sam_image = face_recognition.load_image_file('sam_small.jpeg')
##sam_image_encoding = face_recognition.face_encodings(sam_image)[0]


##match = face_recognition.compare_faces([ashhad_face_encoding], [sam_image_encoding])
##if match[0]:
##    name = "Samrudha Kelkar"
##    print("I see someone named {}!".format(name))
##else:
##    print("Unkown person!")


### Strings 
face_found_string = "A... Hey welcome. to DBS Kiosk la"
let_me_authenticate_string = "A... Let me authenticate you"
could_not_found_string1 = "A. You are not a DBS customer" 
could_not_found_string2 = "Do open. your DBS account."

#qr related
put_qr_string = "Ahh.. Please display the QR"
scanning_qr_image="Ahh... Scanning QR. Please wait"
decoded_qr_data="Ahh... Scanned QR Code. Recipient is "
send_money_qr="Ahh... Sending money to "
send_money_success_qr="Ahh... Money sent succesfully"
thank_visit_again="Ahh... Thank you for visiting DBS Kiosk, visit again"

#gesture related
ask_for_gesture="A.. How was your experience"
thanks_for_review_1="A.. We will try to improve. Sorry for the experience."
thanks_for_review_2="A.. We'll.. do.. better next time."
thanks_for_review_3="A.. Awesome.. Great.. Glad that we could help."


## Capture image for face detection
def getImage():
    camera = picamera.PiCamera()
    camera.resolution = (320, 240)
    output = np.empty((240, 320, 3), dtype=np.uint8)
    camera.start_preview()
    time.sleep(4)
    camera.capture(output, format="rgb")
    camera.close()
    return output

def get_image_for_QR():
  with picamera.PiCamera() as camera:
    camera.resolution = (1024, 768)
    camera.start_preview()
    # Camera warm-up time
    time.sleep(4)
    camera.capture('foo.jpg')
    
def get_usb_camera_image(image):
    subprocess.run(["fswebcam" ,image])
    # cv2.namedWindow('captured_image', cv2.WINDOW_NORMAL)
    # cv2.imshow('image',image)
    # time.sleep(3)
    # cv2.destroyAllWindows()


def match_QR():
    print(put_qr_string)
    subprocess.run(["espeak" ,put_qr_string])
    
    time.sleep(1)
    print ('Taking picture..')
    try:
        f = 1
        #qr_count = len(os.listdir('qr_codes'))
        # get_image_for_QR()
        get_usb_camera_image("foo.jpg")
        #os.system('sudo fswebcam -d /dev/video'+sys.argv[1]+' -q qr_codes/qr_'+str(qr_count)+'.jpg')
        print ('Picture taken..')
    except Exception as e:
        f = 0
        print ('Picture couldn\'t be taken with exception ' + str(e))

    print

    if(f):
        print ('Scanning image..')
        subprocess.run(["espeak" ,scanning_qr_image])
        f = open('foo.jpg','rb')
        qr = Image.open(f);
        qr.load()

        codes = zbarlight.scan_codes('qrcode',qr)
        if(codes==None):
            #os.remove('qr_codes/qr_'+str(qr_count)+'.jpg')
            print ('No QR code found')
            return False
        else:
            print( 'QR code(s):')
            print (codes)
            
            global decoded_qr_data
            global send_money_qr
            print(decoded_qr_data)
            time.sleep(1)

            recepient=str(codes[0],'utf-8')

            decoded_qr_data=decoded_qr_data+recepient
            send_money_qr=send_money_qr+recepient
            
            subprocess.run(["espeak" ,decoded_qr_data])
            time.sleep(3)
            subprocess.run(["espeak" ,send_money_qr])
            time.sleep(3)
            subprocess.run(["espeak" ,send_money_success_qr])
            time.sleep(3)
            # subprocess.run(["espeak" ,thank_visit_again])
            # time.sleep(3)
            return True




#for gesture recognition

def keras_predict(model, image):
    processed = keras_process_image(image)
    pred_probab = model.predict(processed)[0]
    pred_class = list(pred_probab).index(max(pred_probab))
    return max(pred_probab), pred_class


def keras_process_image(img):
    image_x = 50
    image_y = 50
    img = cv2.resize(img, (image_x, image_y))
    img = np.array(img, dtype=np.float32)
    img = np.reshape(img, (-1, image_x, image_y, 1))
    return img

def get_emojis():
    emojis_folder = 'hand_emo/'
    emojis = []
    for emoji in range(len(os.listdir(emojis_folder))):
        print(emoji)
        emojis.append(cv2.imread(emojis_folder+str(emoji)+'.png', -1))
    return emojis


def overlay(image, emoji, x,y,w,h):
    emoji = cv2.resize(emoji, (w, h))
    try:
        image[y:y+h, x:x+w] = blend_transparent(image[y:y+h, x:x+w], emoji)
    except:
        pass
    return image

def blend_transparent(face_img, overlay_t_img):
    # Split out the transparency mask from the colour info
    overlay_img = overlay_t_img[:,:,:3] # Grab the BRG planes
    overlay_mask = overlay_t_img[:,:,3:]  # And the alpha plane

    # Again calculate the inverse mask
    background_mask = 255 - overlay_mask

    # Turn the masks into three channel, so we can use them as weights
    overlay_mask = cv2.cvtColor(overlay_mask, cv2.COLOR_GRAY2BGR)
    background_mask = cv2.cvtColor(background_mask, cv2.COLOR_GRAY2BGR)

    # Create a masked out face image, and masked out overlay
    # We convert the images to floating point in range 0.0 - 1.0
    face_part = (face_img * (1 / 255.0)) * (background_mask * (1 / 255.0))
    overlay_part = (overlay_img * (1 / 255.0)) * (overlay_mask * (1 / 255.0))

    # And finally just add them together, and rescale it back to an 8bit integer image
    return np.uint8(cv2.addWeighted(face_part, 255.0, overlay_part, 255.0, 0.0))



def detectGesture():
    model = load_model('emojinator.h5')
    emojis = get_emojis()
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    x, y, w, h = 300, 50, 350, 350

    while (cap.isOpened()):
        ret, img = cap.read()
        img = cv2.flip(img, 1)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mask2 = cv2.inRange(hsv, np.array([2, 50, 60]), np.array([25, 150, 255]))
        res = cv2.bitwise_and(img, img, mask=mask2)
        gray = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
        median = cv2.GaussianBlur(gray, (5, 5), 0)

        kernel_square = np.ones((5, 5), np.uint8)
        dilation = cv2.dilate(median, kernel_square, iterations=2)
        opening = cv2.morphologyEx(dilation, cv2.MORPH_CLOSE, kernel_square)
        ret, thresh = cv2.threshold(opening, 30, 255, cv2.THRESH_BINARY)

        thresh = thresh[y:y + h, x:x + w]
        contours = cv2.findContours(thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[1]
        if len(contours) > 0:
            contour = max(contours, key=cv2.contourArea)
            if cv2.contourArea(contour) > 2500:
                x, y, w1, h1 = cv2.boundingRect(contour)
                newImage = thresh[y:y + h1, x:x + w1]
                newImage = cv2.resize(newImage, (50, 50))
                pred_probab, pred_class = keras_predict(model, newImage)
                print(pred_class, pred_probab)
                
                #pre display
                img = overlay(img, emojis[pred_class], 400, 250, 90, 90)
                x, y, w, h = 300, 50, 350, 350
                cv2.imshow("Frame", img)
                cv2.imshow("Contours", thresh)

                if int(pred_class)==0:

                    subprocess.run(["espeak" ,thanks_for_review_1])
                    time.sleep(2)
                    subprocess.run(["espeak" ,thanks_for_review_1])
                    time.sleep(4)
                    break
                elif int(pred_class)==1:
                    subprocess.run(["espeak" ,thanks_for_review_2])
                    time.sleep(2)
                    subprocess.run(["espeak" ,thanks_for_review_2])
                    time.sleep(4)
                    break
                elif int(pred_class)==2:
                    subprocess.run(["espeak" ,thanks_for_review_3])
                    time.sleep(2)
                    subprocess.run(["espeak" ,thanks_for_review_3])
                    time.sleep(4)
                    break
                

        

                    

                # img = overlay(img, emojis[pred_class], 400, 250, 90, 90)

        x, y, w, h = 300, 50, 350, 350
        cv2.imshow("Frame", img)
        cv2.imshow("Contours", thresh)
        k = cv2.waitKey(10)
        if k == 27:
            break
    return True
#end of gesture recognition




count = 0

while True:



    

    print("Capturing image.")
    # Grab a single frame of video from the RPi camera as a numpy array
    # output = getImage()

    get_usb_camera_image("face.jpg")

    time.sleep(1)
    cv2.destroyAllWindows()

    output = cv2.imread("face.jpg",cv2.IMREAD_COLOR)


    # f = open('face.jpg','rb')
    # output = Image.open(f);
    # output.load()


    # Find all the faces and face encodings in the current frame of video
    face_locations = face_recognition.face_locations(output)
    print("Found {} faces in image.".format(len(face_locations)))
    face_encodings = face_recognition.face_encodings(output, face_locations)

    # Loop over each face found in the frame to see if it's someone we know.
    for face_encoding in face_encodings:
        # See if the face is a match for the known face(s)

        ##greet msgs
        if count ==0:
            subprocess.run(["espeak" ,face_found_string])
            time.sleep(1)
            subprocess.run(["espeak" ,let_me_authenticate_string])
            time.sleep(1)


        ## match faces
        match = face_recognition.compare_faces([ashhad_face_encoding], face_encoding)
        name = "<Unknown Person>"
        name_string=""

        if match[0]:
            name = "John"
            name_string="A... You must be "+name+". How are you?"
            print("I see someone named {}!".format(name))
            subprocess.run(["espeak" ,name_string])
            time.sleep(1)

            #from here go for QR code match
            qr_match_status = match_QR()
            if qr_match_status:
                print("QR has been matched")
            else:
                print("QR could not be matched")


            detect_gesture_status=False
            if qr_match_status:
                subprocess.run(["espeak" ,ask_for_gesture])
                detect_gesture_status=detectGesture()

            if detect_gesture_status:
                subprocess.run(["espeak" ,thank_visit_again])
                time.sleep(3)
                

                

            
        
        else:

            subprocess.run(["espeak" ,could_not_found_string1])
            time.sleep(1)
            subprocess.run(["espeak" ,could_not_found_string2])
            


        

    time.sleep(1)


