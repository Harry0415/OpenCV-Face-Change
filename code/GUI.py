import cv2
import tkinter as tk
import awesometkinter as atk
import tkinter.font as font
from tkinter import filedialog
from tkinter import messagebox
from PIL import Image, ImageTk

import numpy as np
import dlib

from mtcnn.mtcnn import MTCNN
from keras_vggface.vggface import VGGFace
from keras_vggface.utils import preprocess_input
from keras_vggface.utils import decode_predictions
import PIL


path = []
images = []
img_size = 150
img_path = []

''' 
    ### Decompress the .bz2 files. ###
    
    path = "dlib-models"
    bz2FilenamesList = glob.glob(path + '/*.bz2')
    
    for filename in bz2FilenamesList:
        filepath = path + "/" + filename
        zipfile = bz2.BZ2File(filename)
        data = zipfile.read() 
        newfilepath = filename[:-4]
        print(newfilepath)
        open(newfilepath, 'wb').write(data)
'''
    
def photos_face_swapping():
    
    # Load images and convert to gray scale
    img = cv2.imread(img_path[0])
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Create a mask for target image
    mask = np.zeros_like(img_gray)
    img2 = cv2.imread(img_path[1])
    img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    
    # Get the detecor of face detetion
    detector = dlib.get_frontal_face_detector()
    
    # Load dlib landmark
    predictor = dlib.shape_predictor("../library/dlib-models/shape_predictor_68_face_landmarks.dat")
    height, width, channels = img2.shape
    img2_new_face = np.zeros((height, width, channels), np.uint8)
    
    # Source face
    # Detect the face
    faces = detector(img_gray)
    for face in faces:
        # Use landmark to mark the face by 68 points.
        landmarks = predictor(img_gray, face)
        landmarks_points = []
        for n in range(0, 68):
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            landmarks_points.append((x, y))
        
        # Convexhull the points.
        points = np.array(landmarks_points, np.int32)
        convexhull = cv2.convexHull(points)
        # cv2.polylines(img, [convexhull], True, (255, 0, 0), 3)
        cv2.fillConvexPoly(mask, convexhull, 255)
    
        # Delaunay triangulation
        rect = cv2.boundingRect(convexhull)
        subdiv = cv2.Subdiv2D(rect)
        subdiv.insert(landmarks_points)
        triangles = subdiv.getTriangleList()
        triangles = np.array(triangles, dtype=np.int32)
    
        indexes_triangles = []
        for t in triangles:
            pt1 = (t[0], t[1])
            pt2 = (t[2], t[3])
            pt3 = (t[4], t[5])
    
            index_pt1 = np.where((points == pt1).all(axis=1))
            index_pt1 = extract_index_nparray(index_pt1)
    
            index_pt2 = np.where((points == pt2).all(axis=1))
            index_pt2 = extract_index_nparray(index_pt2)
    
            index_pt3 = np.where((points == pt3).all(axis=1))
            index_pt3 = extract_index_nparray(index_pt3)
    
            if index_pt1 is not None and index_pt2 is not None and index_pt3 is not None:
                triangle = [index_pt1, index_pt2, index_pt3]
                indexes_triangles.append(triangle)
    
    # Destination face
    # Use landmark to mark the face by 68 points.
    faces2 = detector(img2_gray)
    for face in faces2:
        landmarks = predictor(img2_gray, face)
        landmarks_points2 = []
        for n in range(0, 68):
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            landmarks_points2.append((x, y))
    
        points2 = np.array(landmarks_points2, np.int32)
        convexhull2 = cv2.convexHull(points2)
    
    lines_space_mask = np.zeros_like(img_gray)
    # Triangulation of both faces
    for triangle_index in indexes_triangles:
        # Triangulation of the first face
        tr1_pt1 = landmarks_points[triangle_index[0]]
        tr1_pt2 = landmarks_points[triangle_index[1]]
        tr1_pt3 = landmarks_points[triangle_index[2]]
        triangle1 = np.array([tr1_pt1, tr1_pt2, tr1_pt3], np.int32)
    
    
        rect1 = cv2.boundingRect(triangle1)
        (x, y, w, h) = rect1
        cropped_triangle = img[y: y + h, x: x + w]
        cropped_tr1_mask = np.zeros((h, w), np.uint8)
    
    
        points = np.array([[tr1_pt1[0] - x, tr1_pt1[1] - y],
                           [tr1_pt2[0] - x, tr1_pt2[1] - y],
                           [tr1_pt3[0] - x, tr1_pt3[1] - y]], np.int32)
    
        cv2.fillConvexPoly(cropped_tr1_mask, points, 255)
    
        # Lines space
        cv2.line(lines_space_mask, tr1_pt1, tr1_pt2, 255)
        cv2.line(lines_space_mask, tr1_pt2, tr1_pt3, 255)
        cv2.line(lines_space_mask, tr1_pt1, tr1_pt3, 255)
    
        # Triangulation of second face
        tr2_pt1 = landmarks_points2[triangle_index[0]]
        tr2_pt2 = landmarks_points2[triangle_index[1]]
        tr2_pt3 = landmarks_points2[triangle_index[2]]
        triangle2 = np.array([tr2_pt1, tr2_pt2, tr2_pt3], np.int32)
    
    
        rect2 = cv2.boundingRect(triangle2)
        (x, y, w, h) = rect2
    
        cropped_tr2_mask = np.zeros((h, w), np.uint8)
    
        points2 = np.array([[tr2_pt1[0] - x, tr2_pt1[1] - y],
                            [tr2_pt2[0] - x, tr2_pt2[1] - y],
                            [tr2_pt3[0] - x, tr2_pt3[1] - y]], np.int32)
    
        cv2.fillConvexPoly(cropped_tr2_mask, points2, 255)
    
        # Warp triangles
        points = np.float32(points)
        points2 = np.float32(points2)
        M = cv2.getAffineTransform(points, points2)
        warped_triangle = cv2.warpAffine(cropped_triangle, M, (w, h))
        warped_triangle = cv2.bitwise_and(warped_triangle, warped_triangle, mask=cropped_tr2_mask)
    
        # Reconstructing destination face
        img2_new_face_rect_area = img2_new_face[y: y + h, x: x + w]
        img2_new_face_rect_area_gray = cv2.cvtColor(img2_new_face_rect_area, cv2.COLOR_BGR2GRAY)
        _, mask_triangles_designed = cv2.threshold(img2_new_face_rect_area_gray, 1, 255, cv2.THRESH_BINARY_INV)
        warped_triangle = cv2.bitwise_and(warped_triangle, warped_triangle, mask=mask_triangles_designed)
    
        img2_new_face_rect_area = cv2.add(img2_new_face_rect_area, warped_triangle)
        img2_new_face[y: y + h, x: x + w] = img2_new_face_rect_area
    
    
    # Face swapped (putting 1st face into 2nd face)
    img2_face_mask = np.zeros_like(img2_gray)
    img2_head_mask = cv2.fillConvexPoly(img2_face_mask, convexhull2, 255)
    img2_face_mask = cv2.bitwise_not(img2_head_mask)
    
    
    img2_head_noface = cv2.bitwise_and(img2, img2, mask=img2_face_mask)
    result = cv2.add(img2_head_noface, img2_new_face)
    
    (x, y, w, h) = cv2.boundingRect(convexhull2)
    center_face2 = (int((x + x + w) / 2), int((y + y + h) / 2))
    
    seamlessclone = cv2.seamlessClone(result, img2, img2_head_mask, center_face2, cv2.NORMAL_CLONE)
    
    # resize image
    scale_percent = 30 # percent of original size
    width = int(seamlessclone.shape[1] * scale_percent / 100)
    height = int(seamlessclone.shape[0] * scale_percent / 100)
    dim = (width, height)
    seamlessclone = cv2.resize(seamlessclone, dim, interpolation = cv2.INTER_AREA)
    # Save the seamlessclone image.
    cv2.imwrite("../output/seamlessclone.jpg", seamlessclone)
    
    # Load image by use Imgae package and resize.
    im = Image.open("../output/seamlessclone.jpg")
    imTK = ImageTk.PhotoImage(im.resize((250, 300)))
    images.append(imTK)
    
    image_main = tk.Label(div2, image=imTK)
    image_main['width'] = 200
    image_main['height'] = 300
    
    image_main.grid(column=1, row=4, sticky=align_mode)
        
# Extract the array's index
def extract_index_nparray(nparray):
        index = None
        for num in nparray[0]:
            index = num
            break
        return index

# Define the UI layout.
def define_layout(obj, cols=1, rows=1):
    
    def method(trg, col, row):

        for c in range(cols):    
            trg.columnconfigure(c, weight=1)
        for r in range(rows):
            trg.rowconfigure(r, weight=1)

    if type(obj)==list:        
        [method(trg, cols, rows) for trg in obj]
    else:
        trg = obj
        method(trg, cols, rows)
        
# Clean the div2 (Right div).
def clear_frame():
    for widgets in div2.winfo_children():
        widgets.destroy()

# Detfine the homepage content.
def homepage():
    clear_frame()
    label = tk.Label(div2, bg='lightskyblue3', text="Welcome to use the AI face change system.\nPlease select which function you want to do.", font=font.Font(size=23, family='Helvetica')).pack(pady=50)

    label_title1 = tk.Label(div2, bg='DarkOrchid2', text="1.Face switch with photo:", font=font.Font(size=20, family='Helvetica')).pack()
    label_info1 = tk.Label(div2, bg='lightskyblue3', text="You can upload two images to switch\nthe faces from left to right.", font=font.Font(size=15, family='Helvetica')).pack(pady=30)
    
    label_title2 = tk.Label(div2, bg='DarkOrchid2', text="2.Real-time face switch with webcam:", font=font.Font(size=20, family='Helvetica')).pack()
    label_info2 = tk.Label(div2, bg='lightskyblue3', text="You can upload a image to switch\nthe face by webcam.", font=font.Font(size=15, family='Helvetica')).pack(pady=30)
    
    label_title3 = tk.Label(div2, bg='DarkOrchid2', text="3.Detect the similarity of you and celibrity:", font=font.Font(size=20, family='Helvetica')).pack()
    label_info3 = tk.Label(div2, bg='lightskyblue3', text="You can detect the similarity between\nyou and celibrities by upload your image.", font=font.Font(size=15, family='Helvetica')).pack()

# Define the function of upload images.
def UploadAction(co,ro):
    global imTK, images
    filename = filedialog.askopenfilename()
    print('Selected:', filename)
    img_path.append(filename)
    path.append(filename)
    im = Image.open(filename)
    imTK = ImageTk.PhotoImage(im.resize((img_size, img_size)))
    images.append(imTK)
    
    image_main = tk.Label(div2, image=imTK)
    image_main['height'] = img_size
    image_main['width'] = img_size
    
    image_main.grid(column=co, row=ro, sticky=align_mode)

# Define face change div.
def face_change():
    global img_path
    clear_frame()
    img_path = []
    messagebox.showinfo(None, 'Please select source image.')
    button_source = atk.Button3d(div2, text='Source Image', state=tk.DISABLED, command=UploadAction(0,1))
    messagebox.showinfo(None, 'Please select destination image.')     
    button_destination = atk.Button3d(div2, text='Destination Image', state=tk.DISABLED, command=UploadAction(2,1))
    button_change = atk.Button3d(div2, text='Change', command=photos_face_swapping)
    button_source.grid(column=0, row=0, padx=pad, pady=pad, sticky=align_mode)
    button_destination.grid(column=2, row=0, padx=pad, pady=pad, sticky=align_mode)
    button_change.grid(column=1, row=3, padx=pad, pady=pad, sticky=align_mode)

# Real-time face change function.
def show_frame():
    # Open the webcam
    cap = cv2.VideoCapture(0)
    def videoLoop(): 
        _, img2 = cap.read()
        img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        img2_new_face = np.zeros_like(img2)
        
        # Source face
        # Use landmark to mark the face by 68 points.
        faces2 = detector(img2_gray)
        for face in faces2:
            landmarks = predictor(img2_gray, face)
            landmarks_points2 = []
            for n in range(0, 68):
                x = landmarks.part(n).x
                y = landmarks.part(n).y
                landmarks_points2.append((x, y))
    
            # cv2.circle(img2, (x, y), 3, (0, 255, 0), -1)
            points2 = np.array(landmarks_points2, np.int32)
            convexhull2 = cv2.convexHull(points2)
    
    
        # Triangulation of both faces
        for triangle_index in indexes_triangles:
            # Triangulation of the first face
            tr1_pt1 = landmarks_points[triangle_index[0]]
            tr1_pt2 = landmarks_points[triangle_index[1]]
            tr1_pt3 = landmarks_points[triangle_index[2]]
            triangle1 = np.array([tr1_pt1, tr1_pt2, tr1_pt3], np.int32)
    
            rect1 = cv2.boundingRect(triangle1)
            (x, y, w, h) = rect1
            cropped_triangle = faceimg[y: y + h, x: x + w]
            cropped_tr1_mask = np.zeros((h, w), np.uint8)
    
            points = np.array([[tr1_pt1[0] - x, tr1_pt1[1] - y],
                               [tr1_pt2[0] - x, tr1_pt2[1] - y],
                               [tr1_pt3[0] - x, tr1_pt3[1] - y]], np.int32)
    
            cv2.fillConvexPoly(cropped_tr1_mask, points, 255)
    
            # Triangulation of source face
            tr2_pt1 = landmarks_points2[triangle_index[0]]
            tr2_pt2 = landmarks_points2[triangle_index[1]]
            tr2_pt3 = landmarks_points2[triangle_index[2]]
            triangle2 = np.array([tr2_pt1, tr2_pt2, tr2_pt3], np.int32)
            
    
            rect2 = cv2.boundingRect(triangle2)
            (x, y, w, h) = rect2
    
            cropped_tr2_mask = np.zeros((h, w), np.uint8)
    
            points2 = np.array([[tr2_pt1[0] - x, tr2_pt1[1] - y],
                                [tr2_pt2[0] - x, tr2_pt2[1] - y],
                                [tr2_pt3[0] - x, tr2_pt3[1] - y]], np.int32)
    
            cv2.fillConvexPoly(cropped_tr2_mask, points2, 255)
    
    
    
            # Warp triangles
            points = np.float32(points)
            points2 = np.float32(points2)
            M = cv2.getAffineTransform(points, points2)
            warped_triangle = cv2.warpAffine(cropped_triangle, M, (w, h))
            warped_triangle = cv2.bitwise_and(warped_triangle, warped_triangle, mask=cropped_tr2_mask)
    
    
            # Reconstructing destination face
            img2_new_face_rect_area = img2_new_face[y: y + h, x: x + w]
            img2_new_face_rect_area_gray = cv2.cvtColor(img2_new_face_rect_area, cv2.COLOR_BGR2GRAY)
            _, mask_triangles_designed = cv2.threshold(img2_new_face_rect_area_gray, 1, 255, cv2.THRESH_BINARY_INV)
            warped_triangle = cv2.bitwise_and(warped_triangle, warped_triangle, mask=mask_triangles_designed)
    
            img2_new_face_rect_area = cv2.add(img2_new_face_rect_area, warped_triangle)
            img2_new_face[y: y + h, x: x + w] = img2_new_face_rect_area
    
    
        # Face swapped (putting 1st face into 2nd face)
        img2_face_mask = np.zeros_like(img2_gray)
        img2_head_mask = cv2.fillConvexPoly(img2_face_mask, convexhull2, 255)
        img2_face_mask = cv2.bitwise_not(img2_head_mask)
    
    
        img2_head_noface = cv2.bitwise_and(img2, img2, mask=img2_face_mask)
        result = cv2.add(img2_head_noface, img2_new_face)
    
        (x, y, w, h) = cv2.boundingRect(convexhull2)
        center_face2 = (int((x + x + w) / 2), int((y + y + h) / 2))
    
        seamlessclone = cv2.seamlessClone(result, img2, img2_head_mask, center_face2, cv2.MIXED_CLONE)
        seamlessclone = cv2.cvtColor(seamlessclone, cv2.COLOR_BGR2RGB)
        
        img = Image.fromarray(seamlessclone)
        imgtk = ImageTk.PhotoImage(image=img)
        lmain.imgtk = imgtk
        lmain.configure(image=imgtk)
        lmain.after(1, videoLoop) 
    
    global img_path
    clear_frame()
    img_path = []
    clear_frame()
    messagebox.showinfo(None, 'Please select source image.')    
    button_source = atk.Button3d(div2, text='Source Image', state=tk.DISABLED, command=UploadAction(0,1))
    lmain = tk.Label(div2)
    lmain.grid()
    
    # Load the image and convert to gray scale.
    faceimg = cv2.imread(img_path[0])
    img_gray = cv2.cvtColor(faceimg, cv2.COLOR_BGR2GRAY)
    # Create a mask for target image
    mask = np.zeros_like(img_gray)
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("../library/dlib-models/shape_predictor_68_face_landmarks.dat")
    
    indexes_triangles = []
    
    # Destination face
    # Use landmark to mark the face by 68 points.
    faces = detector(img_gray)
    for face in faces:
        landmarks = predictor(img_gray, face)
        landmarks_points = []
        for n in range(0, 68):
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            landmarks_points.append((x, y))
    
            # cv2.circle(img, (x, y), 3, (0, 0, 255), -1)
    
        points = np.array(landmarks_points, np.int32)
        convexhull = cv2.convexHull(points)
        # cv2.polylines(img, [convexhull], True, (255, 0, 0), 3)
        cv2.fillConvexPoly(mask, convexhull, 255)
    
        # Delaunay triangulation
        rect = cv2.boundingRect(convexhull)
        subdiv = cv2.Subdiv2D(rect)
        subdiv.insert(landmarks_points)
        triangles = subdiv.getTriangleList()
        triangles = np.array(triangles, dtype=np.int32)
    
        indexes_triangles = []
        for t in triangles:
            pt1 = (t[0], t[1])
            pt2 = (t[2], t[3])
            pt3 = (t[4], t[5])
    
            index_pt1 = np.where((points == pt1).all(axis=1))
            index_pt1 = extract_index_nparray(index_pt1)
    
            index_pt2 = np.where((points == pt2).all(axis=1))
            index_pt2 = extract_index_nparray(index_pt2)
    
            index_pt3 = np.where((points == pt3).all(axis=1))
            index_pt3 = extract_index_nparray(index_pt3)
    
            if index_pt1 is not None and index_pt2 is not None and index_pt3 is not None:
                triangle = [index_pt1, index_pt2, index_pt3]
                indexes_triangles.append(triangle)
    
    videoLoop()

def detect_celibretiy_simularity():
    # Load images and convert to gray scale
    img = cv2.imread(img_path[0])
    img = np.asarray(img, dtype='uint8')
    
    detector = MTCNN()
    
    # set face extraction parameters
    border_rel = 0 # increase or decrease zoom on image
    
    # detect faces in the image
    detections = detector.detect_faces(img)
    print(detections)
    
    x1, y1, width, height = detections[0]['box']
    dw = round(width * border_rel)
    dh = round(height * border_rel)
    x2, y2 = x1 + width + dw, y1 + height + dh
    face = img[y1:y2, x1:x2]
    
    # resize pixels to the model size
    face = PIL.Image.fromarray(face)
    face = face.resize((224, 224))
    face = np.asarray(face)
    
    # convert to float32
    face_pp = face.astype('float32')
    face_pp = np.expand_dims(face_pp, axis = 0)
    
    face_pp = preprocess_input(face_pp, version = 2)
    
    # Create the resnet50 Model
    model = VGGFace(model= 'resnet50')
    # Check what the required input of the model is & output
    print('Inputs: {input}'.format(input = model.inputs))
    print('Output: {output}'.format(output = model.outputs))
    
    # predict the face with the input
    prediction = model.predict(face_pp)
    
    # convert predictions into names & probabilities
    results = decode_predictions(prediction)
    
    outputresult = "Result:\n"
    for result in results[0]:
        # print('%s: %.3f%%' % (result[0], result[1]*100))
        outputresult += ('%s: %.3f%%\n' % (result[0], result[1]*100))
    label_output = tk.Label(div2, bg='lightskyblue3', text=outputresult, justify='left')
    label_output.grid(column=2, row=25, padx=pad, pady=pad, sticky=align_mode)
        
def detect_simularity():
    global img_path
    img_path = []
    clear_frame()
    messagebox.showinfo(None, 'Please select image.')
    UploadAction(0,25)
    button_detect_simularity = atk.Button3d(div2, text='Detect', command=detect_celibretiy_simularity)
    button_detect_simularity.grid(column=1, row=30, padx=pad, pady=pad, sticky=align_mode)


### Design the default UI ###
root = tk.Tk()
root.title('AI face change system')
align_mode = 'nswe'
pad = 5
fontsize = font.Font(size=20, family='Helvetica')

root.geometry('1100x700')
root.resizable(False, False)

div1 = tk.Frame(root, bg='skyblue3')
div2 = tk.Frame(root, bg='lightskyblue3')

div1.grid(column=0, row=0, padx=pad, pady=pad, columnspan=1, sticky=align_mode)
div2.grid(column=1, row=0, padx=pad, pady=pad, columnspan=4, sticky=align_mode)

define_layout(root, cols=5)
define_layout([div1, div2])

label = tk.Label(div2, bg='lightskyblue3', text="Welcome to use the AI face change system.\nPlease select which function you want to do.", font=font.Font(size=23, family='Helvetica')).pack(pady=50)

label_title1 = tk.Label(div2, bg='DarkOrchid2', text="1.Face switch with photo:", font=font.Font(size=20, family='Helvetica')).pack()
label_info1 = tk.Label(div2, bg='lightskyblue3', text="You can upload two images to switch\nthe faces from left to right.", font=font.Font(size=15, family='Helvetica')).pack(pady=30)

label_title2 = tk.Label(div2, bg='DarkOrchid2', text="2.Real-time face switch with webcam:", font=font.Font(size=20, family='Helvetica')).pack()
label_info2 = tk.Label(div2, bg='lightskyblue3', text="You can upload a image to switch\nthe face by webcam.", font=font.Font(size=15, family='Helvetica')).pack(pady=30)

label_title3 = tk.Label(div2, bg='DarkOrchid2', text="3.Detect the similarity of you and celibrity:", font=font.Font(size=20, family='Helvetica')).pack()
label_info3 = tk.Label(div2, bg='lightskyblue3', text="You can detect the similarity between\nyou and celibrities by upload your image.", font=font.Font(size=15, family='Helvetica')).pack(pady=30)

button_home = tk.Button(div1, bg='gray', text='Home Page', font=fontsize, width=15, command=homepage)
button_face_change = tk.Button(div1, bg='gray', text='Face switch with photo', font=fontsize,  width=15, command=face_change)
button_webcam = tk.Button(div1, bg='gray', text='Real-time face switch\nwith webcam', font=fontsize, width=15, command=show_frame)
button_detect_simularity = tk.Button(div1, bg='gray', text='Detect the similarity\nof you and celibrity', font=fontsize, width=15, command=detect_simularity)
button_home.grid(column=0, row=0, padx=pad, pady=50, sticky=align_mode)
button_face_change.grid(column=0, row=1, padx=pad, pady=50, sticky=align_mode)
button_webcam.grid(column=0, row=2, padx=pad, pady=50, sticky=align_mode)
button_detect_simularity.grid(column=0, row=3, padx=pad, pady=50, sticky=align_mode)

define_layout(root, cols=5)
define_layout(div1, rows=4)
define_layout(div2, cols=3, rows=50)

root.mainloop()