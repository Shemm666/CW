# pylint: disable=invalid-name, redefined-outer-name, missing-docstring, non-parent-init-called, trailing-whitespace, line-too-long
import cv2
import numpy as np
import cv2, threading, time
import queue as Queue
from keras.models import model_from_json
from threading import Thread
from datetime import datetime as dt
import datetime
import re
import shutil

class Label:
    def __init__(self, cl=-1, tl=np.array([0., 0.]), br=np.array([0., 0.]), prob=None):
        self.__tl = tl
        self.__br = br
        self.__cl = cl
        self.__prob = prob

    def __str__(self):
        return 'Class: %d, top left(x: %f, y: %f), bottom right(x: %f, y: %f)' % (
        self.__cl, self.__tl[0], self.__tl[1], self.__br[0], self.__br[1])

    def copy(self):
        return Label(self.__cl, self.__tl, self.__br)

    def wh(self): return self.__br - self.__tl

    def cc(self): return self.__tl + self.wh() / 2

    def tl(self): return self.__tl

    def br(self): return self.__br

    def tr(self): return np.array([self.__br[0], self.__tl[1]])

    def bl(self): return np.array([self.__tl[0], self.__br[1]])

    def cl(self): return self.__cl

    def area(self): return np.prod(self.wh())

    def prob(self): return self.__prob

    def set_class(self, cl):
        self.__cl = cl

    def set_tl(self, tl):
        self.__tl = tl

    def set_br(self, br):
        self.__br = br

    def set_wh(self, wh):
        cc = self.cc()
        self.__tl = cc - .5 * wh
        self.__br = cc + .5 * wh

    def set_prob(self, prob):
        self.__prob = prob

class DLabel(Label):
    def __init__(self, cl, pts, prob):
        self.pts = pts
        tl = np.amin(pts, axis=1)
        br = np.amax(pts, axis=1)
        Label.__init__(self, cl, tl, br, prob)

def getWH(shape):
    return np.array(shape[1::-1]).astype(float)

def IOU(tl1, br1, tl2, br2):
    wh1, wh2 = br1-tl1, br2-tl2
    assert((wh1 >= 0).all() and (wh2 >= 0).all())
    
    intersection_wh = np.maximum(np.minimum(br1, br2) - np.maximum(tl1, tl2), 0)
    intersection_area = np.prod(intersection_wh)
    area1, area2 = (np.prod(wh1), np.prod(wh2))
    union_area = area1 + area2 - intersection_area
    return intersection_area/union_area

def IOU_labels(l1, l2):
    return IOU(l1.tl(), l1.br(), l2.tl(), l2.br())

def nms(Labels, iou_threshold=0.5):
    SelectedLabels = []
    Labels.sort(key=lambda l: l.prob(), reverse=True)
    
    for label in Labels:
        non_overlap = True
        for sel_label in SelectedLabels:
            if IOU_labels(label, sel_label) > iou_threshold:
                non_overlap = False
                break

        if non_overlap:
            SelectedLabels.append(label)
    return SelectedLabels



def find_T_matrix(pts, t_pts):
    A = np.zeros((8, 9))
    for i in range(0, 4):
        xi = pts[:, i]
        xil = t_pts[:, i]
        xi = xi.T
        
        A[i*2, 3:6] = -xil[2]*xi
        A[i*2, 6:] = xil[1]*xi
        A[i*2+1, :3] = xil[2]*xi
        A[i*2+1, 6:] = -xil[0]*xi

    [U, S, V] = np.linalg.svd(A)
    H = V[-1, :].reshape((3, 3))
    return H

def getRectPts(tlx, tly, brx, bry):
    return np.matrix([[tlx, brx, brx, tlx], [tly, tly, bry, bry], [1, 1, 1, 1]], dtype=float)

def normal(pts, side, mn, MN):
    pts_MN_center_mn = pts * side
    pts_MN = pts_MN_center_mn + mn.reshape((2, 1))
    pts_prop = pts_MN / MN.reshape((2, 1))
    return pts_prop

# Reconstruction function from predict value into plate crpoped from image
def reconstruct(I, Iresized, Yr, lp_threshold):
    # 4 max-pooling layers, stride = 2
    net_stride = 2**4
    side = ((208 + 40)/2)/net_stride

    # one line and two lines license plate size
    one_line = (470, 110)
    two_lines = (280, 200)

    Probs = Yr[..., 0]
    Affines = Yr[..., 2:]

    xx, yy = np.where(Probs > lp_threshold)
    # CNN input image size 
    WH = getWH(Iresized.shape)
    # output feature map size
    MN = WH/net_stride

    vxx = vyy = 0.5 #alpha
    base = lambda vx, vy: np.matrix([[-vx, -vy, 1], [vx, -vy, 1], [vx, vy, 1], [-vx, vy, 1]]).T
    labels = []
    labels_frontal = []

    for i in range(len(xx)):
        x, y = xx[i], yy[i]
        affine = Affines[x, y]
        prob = Probs[x, y]

        mn = np.array([float(y) + 0.5, float(x) + 0.5])

        # affine transformation matrix
        A = np.reshape(affine, (2, 3))
        A[0, 0] = max(A[0, 0], 0)
        A[1, 1] = max(A[1, 1], 0)
        # identity transformation
        B = np.zeros((2, 3))
        B[0, 0] = max(A[0, 0], 0)
        B[1, 1] = max(A[1, 1], 0)

        pts = np.array(A*base(vxx, vyy))
        pts_frontal = np.array(B*base(vxx, vyy))

        pts_prop = normal(pts, side, mn, MN)
        frontal = normal(pts_frontal, side, mn, MN)

        labels.append(DLabel(0, pts_prop, prob))
        labels_frontal.append(DLabel(0, frontal, prob))
        
    final_labels = nms(labels, 0.1)
    final_labels_frontal = nms(labels_frontal, 0.1)

    #print(final_labels_frontal)

    # LP size and type
    out_size, lp_type = (two_lines, 2) if ((final_labels_frontal[0].wh()[0] / final_labels_frontal[0].wh()[1]) < 1.7) else (one_line, 1)

    TLp = []
    Cor = []
    if len(final_labels):
        final_labels.sort(key=lambda x: x.prob(), reverse=True)
        for _, label in enumerate(final_labels):
            t_ptsh = getRectPts(0, 0, out_size[0], out_size[1])
            ptsh = np.concatenate((label.pts * getWH(I.shape).reshape((2, 1)), np.ones((1, 4))))
            H = find_T_matrix(ptsh, t_ptsh)
            Ilp = cv2.warpPerspective(I, H, out_size, borderValue=0)
            TLp.append(Ilp)
            Cor.append(ptsh)
    return final_labels, TLp, lp_type, Cor



import cv2, threading, time
import queue as Queue

# bufferless VideoCapture

def detect_lp(model, I, max_dim, lp_threshold):
    	min_dim_img = min(I.shape[:2])
    	factor = float(max_dim) / min_dim_img
    	w, h = (np.array(I.shape[1::-1], dtype=float) * factor).astype(int).tolist()
    	Iresized = cv2.resize(I, (w, h))
    	T = Iresized.copy()
    	T = T.reshape((1, T.shape[0], T.shape[1], T.shape[2]))
    	Yr = model.predict(T)
    	Yr = np.squeeze(Yr)
    	#print(Yr.shape)
    	L, TLp, lp_type, Cor = reconstruct(I, Iresized, Yr, lp_threshold)
    	return L, TLp, lp_type, Cor

class VideoCapture:

    def __init__(self, name):
        self.cap = cv2.VideoCapture(name)
        self.q = Queue.Queue()
        t = threading.Thread(target=self._reader)
        t.daemon = True
        t.start()

    # read frames as soon as they are available, keeping only most recent one
    def _reader(self, size=(1280, 720)):
        while True:
            time.sleep(0.01)
            ret, frame = self.cap.read()
            #if not ret:
             #   self.cap.set(cv2.CAP_PROP_POS_FRAMES,1)
            if not self.q.empty():
                try:
                    self.q.get_nowait()   # discard previous (unprocessed) frame
                except Queue.Empty:
                    pass
            
            self.q.put(cv2.resize(frame, size))

    def read(self):
            
        return self.q.get()
        
class vehile_dnn_detector:
    def __init__(self, size=(300,300),\
                 net = cv2.dnn.readNetFromCaffe('MobileNetSSD_deploy.prototxt.txt','MobileNetSSD_deploy.caffemodel')):
        self.net=net
        self.size=size
        self.class_=None
        self.prob=None
        self.startX=None
        self.startY=None
        self.endX=None
        self.endY=None
        self.detected=False
    def detect(self,frame,thresh=0.25, draw_countur=False):        
        blob = cv2.dnn.blobFromImage(frame, size=(300, 300), ddepth=cv2.CV_8U)
        self.net.setInput(blob, scalefactor=1.0/127.5, mean=[127.5, 127.5, 127.5])
        det = self.net.forward()
        #метки классов bus,car,motorbike-6,7,14 соответсвенно
        detected=[(class_, probability, coord) for class_, probability, coord\
          in zip(det[0,0,:,1], det[0,0,:,2], det[0,0,:,3:7]) if ((class_ in [6,7,14]) and (probability>thresh))]
        (H, W) = frame.shape[:2]
        try:
            max_prob=max(detected, key = lambda i : i[1])
            box = max_prob[2] * np.array([W, H, W, H])
            (self.startX, self.startY, self.endX, self.endY) = np.abs(box.astype("int") )
            self.detected=True
            if draw_countur:
                cv2.rectangle(img, (startX, startY), (endX, endY),(0, 255, 0), 2)
        except ValueError:
            max_prob=(None, None, [None, None, None, None])
            (self.startX, self.startY, self.endX, self.endY)=max_prob[2]
        self.class_=max_prob[0]
        self.prob=max_prob[1]
               
        return (self.startX, self.startY, self.endX, self.endY)
    
    def centroid(self):
        x=self.startX/2+self.endX/2
        y=self.startY/2+self.endY/2
        return int(x), int(y)
        
class VehInBoxCassif:
    def __init__(self, model_json, model_weights):
        json_file = open(model_json, 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        self.model = model_from_json(loaded_model_json)        
        self.model.load_weights(model_weights)
    def predict_out(self, img, thresh):
        img=img/255
        
        img=cv2.resize(img,(240,240))
        X=img[np.newaxis, ..., np.newaxis]
        return self.model.predict(X)<thresh 
        
class NumbSaver:
    def __init__(self):
        self.numb_list=[]
        
    def numb_add(self,text):        
    	
        text=text.translate(str.maketrans({'|':'I', ' ':'', '-':'', 'i':'I'}))
        text=re.sub('(\d{4})*1(\D{0,1}\d)[I1]{0,1}$','\\1I\\2',text.strip())
        text=re.sub('(\d{4})*1(\D{1}\d)[I1]{0,1}$','\\1I\\2',text)
        text1=re.search(r'\d{4}[A-Za-z]{2}\d{1}',text)
        text2=re.search(r'\d{1}[A-Za-z]{3}\d{4}',text)
        text3=re.search(r'[A-Za-z]{2}\d{5}',text)
        text4=re.search(r'\d{4}[A-Za-z]{2}',text)
        if text1:
            self.numb_list.append(text1.group(0))
        elif text2:
            self.numb_list.append(text2.group(0))
        elif text3:
            self.numb_list.append(text3.group(0))
        elif text4:
            self.numb_list.append(text4.group(0))
    
    def most_frequent(self):
        return max(set(self.numb_list), key = self.numb_list.count)

class ImageSaver:
    '''
forms a list of images with an interval of ms
'''
    def __init__(self,gap):
        self.img_list=[]
        self.gap=gap
        self.last_time=None
        self.stop=False
    
    def save_img(self, img):
        if self.last_time==None and not self.stop:
            self.img_list.append(img)
            self.last_time=dt.now()
            print(1)
        elif (dt.now()-self.last_time).microseconds>self.gap*1000 and not self.stop:
            self.img_list.append(img)
            self.last_time=dt.now()

class LpDetector:
    def __init__(self):       
        json_file=open('wpod-net.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        self.model = model_from_json(loaded_model_json, custom_objects={})
        self.model.load_weights('wpod-net.h5')
        
    def get_plate(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img / 255
        Dmax = 900
        Dmin = 470        
        ratio = float(max(img.shape[:2])) / min(img.shape[:2])
        side = int(ratio * Dmin)
        bound_dim = min(side, Dmax)
        _ , LpImg, _, cor = self.detect_lp(self.model, img, bound_dim, lp_threshold=0.2)        
        return LpImg, cor
    
    def detect_lp(self, model, I, max_dim, lp_threshold):
    	min_dim_img = min(I.shape[:2])
    	factor = float(max_dim) / min_dim_img
    	w, h = (np.array(I.shape[1::-1], dtype=float) * factor).astype(int).tolist()
    	Iresized = cv2.resize(I, (w, h))
    	T = Iresized.copy()
    	T = T.reshape((1, T.shape[0], T.shape[1], T.shape[2]))
    	Yr = model.predict(T)
    	Yr = np.squeeze(Yr)
    	#print(Yr.shape)
    	L, TLp, lp_type, Cor = reconstruct(I, Iresized, Yr, lp_threshold)
    	return L, TLp, lp_type, Cor
    
        
class WebcamVideoStream:
    def __init__(self, src=0, name="WebcamVideoStream"):
        # initialize the video camera stream and read the first frame
        # from the stream
        self.stream = cv2.VideoCapture(src)
        (self.grabbed, self.frame) = self.stream.read()

        # initialize the thread name
        self.name = name

        # initialize the variable used to indicate if the thread should
        # be stopped
        self.stopped = False

    def start(self):
        # start the thread to read frames from the video stream
        t = Thread(target=self.update, name=self.name, args=())
        t.daemon = True
        t.start()
        return self

    def update(self):
        
        # keep looping infinitely until the thread is stopped
        while True:
            cv2.waitKey(1)
            # if the thread indicator variable is set, stop the thread

            if self.stopped:
                return

            # otherwise, read the next frame from the stream
            (self.grabbed, self.frame) = self.stream.read()
            sleep(0.01)
    def read(self):
        
        # return the frame most recently read
        return self.frame

    def stop(self):
        # indicate that the thread should be stopped
        self.stopped = True

class motion_detector:
    '''Ищет разницу на двух последовательных
    изображениях в сером, возвращает контуры движещихся объектов
    площадью выше thresh'''
    def __init__(self, thresh):
        self.last_img=None
        self.thresh=thresh
        self.sleep=False
        self.cnts=[]
    def do_detect(self, gray, video_out=False):
        if not self.sleep:
            if self.last_img is None:
                self.last_img=gray
            frameDelta = cv2.absdiff(self.last_img, gray)
            thresh_img = cv2.threshold(frameDelta, 25, 255, cv2.THRESH_BINARY)[1]
            thresh_img = cv2.dilate(thresh_img, None, iterations=5)
            cnts = cv2.findContours(thresh_img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
            self.last_img=gray
            self.cnts=[cnt for cnt in cnts if cv2.contourArea(cnt)>self.thresh]
            if video_out:            
                for c in self.cnts:
                    (x, y, w, h) = cv2.boundingRect(c)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    
        else: self.cnts=[]    
                
def img_to_recogn(plate_img_list):
    plate_image=cv2.convertScaleAbs(plate_img_list, alpha=(255.0))
    plate_image = cv2.fastNlMeansDenoisingColored(plate_image,None,10,10,7,21)
    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    plate_image = cv2.filter2D(plate_image, -1, kernel)
    gray = cv2.cvtColor(plate_image, cv2.COLOR_BGR2GRAY)
    gray=cv2.equalizeHist(gray)
    gray = cv2.GaussianBlur(gray, (23,23), 0)
    kernel = np.ones((2,2),np.uint8)
    gray = cv2.erode(gray,kernel,iterations = 1)    
    return gray

def img_treat(img,scale=1):
    '''Предобработка изображения'''
    img = cv2.resize(img, (0,0), fx=scale, fy=scale,interpolation=cv2.INTER_LINEAR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #convert to grey scale
    gray = cv2.bilateralFilter(gray, 11, 17, 17) #Blur to reduce noise
    return gray
    
class SampleWriter:
    def __init__(self, delta):
        self.start_at=dt.now()
        self.delta=delta
    def write(self, img, directory):
        if (dt.now()-self.start_at).total_seconds()>self.delta:
            cv2.imwrite(directory+str(dt.now().strftime('%y_%m_%d %H_%M'))+'.jpg', img)
            self.start_at=dt.now()
            
def old_m_d(directory, delta):
    '''delete foders in directory with create date > delta'''
    dir_list=os.listdir(directory)
    for f in dir_list:
        path=directory+'/'+f
        create_date=dt.fromtimestamp(os.stat(path)[-1])
        if (dt.now()-create_date).days>delta:
            os.remove(path) if os.path.isfile(path) else shutil.rmtree(path)


