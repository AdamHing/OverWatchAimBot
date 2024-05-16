import cv2
import numpy as np
from mss import mss
from ultralytics import YOLO
import time 
# import fastgrab
import win32api, win32con
import pydirectinput as pdi
import keyboard


import keyboard
import ctypes
# C struct redefinitions 
PUL = ctypes.POINTER(ctypes.c_ulong)
class KeyBdInput(ctypes.Structure):
    _fields_ = [("wVk", ctypes.c_ushort),
                ("wScan", ctypes.c_ushort),
                ("dwFlags", ctypes.c_ulong),
                ("time", ctypes.c_ulong),
                ("dwExtraInfo", PUL)]
 
class HardwareInput(ctypes.Structure):
    _fields_ = [("uMsg", ctypes.c_ulong),
                ("wParamL", ctypes.c_short),
                ("wParamH", ctypes.c_ushort)]
 
class MouseInput(ctypes.Structure):
    _fields_ = [("dx", ctypes.c_long),
                ("dy", ctypes.c_long),
                ("mouseData", ctypes.c_ulong),
                ("dwFlags", ctypes.c_ulong),
                ("time",ctypes.c_ulong),
                ("dwExtraInfo", PUL)]
 
class Input_I(ctypes.Union):
    _fields_ = [("ki", KeyBdInput),
                 ("mi", MouseInput),
                 ("hi", HardwareInput)]
 
class Input(ctypes.Structure):
    _fields_ = [("type", ctypes.c_ulong),
                ("ii", Input_I)]
 
def MouseMoveTo(x, y):
    extra = ctypes.c_ulong(0)
    ii_ = Input_I()
    ii_.mi = MouseInput(x, y, 0, 0x0001, 0, ctypes.pointer(extra))
 
    command = Input(ctypes.c_ulong(0), ii_)
    ctypes.windll.user32.SendInput(1, ctypes.pointer(command), ctypes.sizeof(command))
 

width = win32api.GetSystemMetrics(0)
height = win32api.GetSystemMetrics(1)
print(width, height)

width = 1920
height = 1080

# width = 800
# height = 800
# x = -10
# y = 500
# win32api.mouse_event(win32con.MOUSEEVENTF_MOVE, x, y, 0, 0)


bounding_box= {'top': 0, 'left': 0, 'width': width, 'height': height }
sct = mss()

# # Load a model
model = YOLO(r'D:\Coding\Projects\OverWatchCV\runs\detect\train24\weights\best.pt')
# # Run batched inference on a list of images
# results = model("screen")  # return a list of Results objects

last_time = time.time()
x = 0
s = time.time()
while 1:
    x+=1
    # img = np.asarray(sct.grab(bounding_box))   
    # imS = cv2.resize(img, (win32api.GetSystemMetrics(0), win32api.GetSystemMetrics(1)))
    imS = np.array(sct.grab(bounding_box))
    imS = cv2.cvtColor(imS,cv2.COLOR_RGBA2RGB)

    # Run YOLO inference on the screen capture
    #source = cv2.imread(r"D:\Coding\Projects\OverWatchCV\overwatch2-1\valid\images\-2022-05-20-17-12-19_mp4-242_jpg.rf.6e61a5bffdee6ecec834761f7110a619.jpg")
    
    # print(source.shape)
    # print(source)
  
    #inference
    results = model.track(imS,persist=True,max_det=2)
    #results = model(imS)
    
    # Visualize the detections
    annotated_image = results[0].plot()
    annotated_image = cv2.resize(annotated_image,(900,900))
    cv2.imshow("screen", annotated_image)

    boxes = results[0].boxes.xyxy.tolist()
    classes = results[0].boxes.cls.tolist()
    names = results[0].names
    confidences = results[0].boxes.conf.tolist()
    
    xt=[]
    yt=[]
   
    for box, cls, conf in zip(boxes, classes, confidences):
        x1, y1, x2, y2 = box
        confidence = conf
        detected_class = cls
        name = names[int(cls)]
        
        #midpoint
        xt.append(abs(int(round((x1+x2)/2))))
        yt.append(abs(int(round((y1+y2)/2))))
    
    
    clamp = lambda n, minn, maxn: max(min(maxn, n), minn)
    
    if len(xt)>0:
        print(xt[0],yt[0])
        
        X = xt[0]-960
        Y = yt[0]-540
        X = clamp(X, -100, 100)
        Y = clamp(Y, -100, 100)
        
        # MouseMoveTo(X,Y)
        
    
        # pdi.moveTo(xt, yt)
        #win32api.mouse_event(win32con.MOUSEEVENTF_MOVE, xt, yt, 0, 0)
        
    if keyboard.is_pressed('alt'):
            break

    if cv2.waitKey(1) & 0xFF == ord("q"):
        cv2.destroyAllWindows()
        break
    
e = time.time()

print("frames",x)
print("time",e-s)
print("fps",x/(e-s))
    
    
#r'D:\Coding\Projects\OverWatchCV\overwatch2-1\valid\images\SRM_hanamura_mp4-138_jpg.rf.78219312db4571f87927eab9b1ff49c7.jpg'