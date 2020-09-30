#! / usr / bin / env opencv
import configparser
import telepot
token = '1227225364:AAGUPkJGl4FVHmhLvtW1Sbh10rKycM1mR8I'
config = configparser.ConfigParser()
config.read("conf.ini")
import cv2
import numpy as np
import pytesseract
pytesseract.pytesseract.tesseract_cmd = config['tesseract']['path_to_tess']
custom_config=config['tesseract']['tess_config']
from datetime import datetime as dt

#from imutils.video import WebcamVideoStream
from utils import  vehile_dnn_detector, VehInBoxCassif, motion_detector, LpDetector, img_to_recogn, img_treat, ImageSaver, NumbSaver, SampleWriter
import re
from time import sleep

from utils import VideoCapture
import os
from sql_utils import postgree  
from write_test import write_test_img  
import sys



point=config['point']['point']
detector=vehile_dnn_detector()
motion=motion_detector(int(config['motion_det']['motion_det_thresh_area']))
lp_det=LpDetector()
img_saver=ImageSaver(200)
out_class=VehInBoxCassif(config['cnn_out_in']['model_json'], config['cnn_out_in']['model_weights']) 
sql_query="""INSERT INTO Time (auto_numb, time_in, time_out) VALUES (%s, %s, %s)"""
sample_writer=SampleWriter(1200)
is_starter=True
h=0
auto_in=False
my1, my2, mx1, mx2=[int(i) for i in config['motion_det']['motion_det'].split(',')]
dy1, dy2, dx1, dx2=[int(i) for i in config['vehicle_det']['vehicle_dnn_area'].split(',')]
py1, py2, px1, px2=[int(i) for i in config['plate_det']['plate_det_area'].split(',')]
oy1, oy2, ox1, ox2=[int(i) for i in config['in_out_classif']['out_pred_area'].split(',')]
TelegramBot = telepot.Bot(token)

channel=int(sys.argv[1])
import os
os.makedirs('images_to_recogn', exist_ok=True)
os.makedirs('images_to_recogn/'+str(channel), exist_ok=True)
os.makedirs('test', exist_ok=True)
while h==0:
    try:
        
        #cap =  WebcamVideoStream(src="rtsp://test:Test1234@128.65.48.128:8079/cam/realmonitor?channel={}&subtype=0".format(str(channel))).start()
        cap =  VideoCapture(config['flow']['path_to_flow'].format(str(channel)))
        #cap=VideoCapture('/home/user/CW/video.mp4')
        frame = cap.read()
        h,w=frame.shape[0],frame.shape[1]
        
        
    except AttributeError:
        pass
try:
    while True:


        frame = cap.read()        
        gray=img_treat(frame)
        sample_writer.write(frame, 'CW/sample/')
        motion.do_detect(gray[my1:my2, mx1:mx2],False)
        
        if motion.cnts:
            now=dt.now()
            day=now.day
            hour=now.hour
            minute=now.minute
            


            startX, startY, endX, endY=detector.detect(frame[dy1:dy2, dx1:dx2],thresh=float(config['vehicle_det']['vehicle_dnn_thresh']))
            if all([startX, startY, endX, endY]):      

                is_starter=True
                print('in')
                print(now)
                motion.sleep=True
                motion.cnts=[]
                auto_in_time=dt.now()
                auto_in=True
                img_saver.img_list=[]



        if auto_in and is_starter:
            img_saver.save_img(frame)
            if (dt.now()-now).total_seconds()>5:
                path='images_to_recogn/'+str(channel)+now.strftime("%m_%d %H_%M_%S")
                os.mkdir(path)
                saved_numb=NumbSaver()
                print(len(img_saver.img_list))
                print(dt.now())
                for i, img in enumerate(img_saver.img_list):

                    try:
                        plate_img_list, plate_img_cnts=lp_det.get_plate(img[py1:py2, px1:px2])
                    except IndexError:
                        plate_img_list=None
                        text=''
                    if plate_img_list:
                        for plate_img in plate_img_list:                    
                            text = pytesseract.image_to_string(img_to_recogn(plate_img), lang='eng',config=custom_config)
                            saved_numb.numb_add(text)
                    cv2.imwrite(os.path.join(path, str(i)+text+'.jpg'),img)
                try:
                    text=saved_numb.most_frequent()
                except ValueError:
                    text='not recogn'
                print(text)
                print(saved_numb.numb_list)
                print(dt.now())
                is_starter=False       
                continue

        elif auto_in and not is_starter:        
            if out_class.predict_out(gray[oy1:oy2,ox1:ox2], float(config['in_out_classif']['out_pred_thresh'])):
                auto_in=False
                write_test_img(frame, path=config['tests']['test_img_path'],fname='test')
                #postgree(eval(config['sql']['sql_conn_params']),sql_query,(text,now, dt.now()))
                print(text, now, dt.now())

                motion.sleep=False

                print('out')
                print(dt.now())
            sleep(2)

        else:
            sleep(0.5)

        cv2.putText(frame,str(auto_in), (10,80), cv2.FONT_HERSHEY_SIMPLEX, 2, 255)

        cv2.imshow('b',frame)
        key=cv2.waitKey(1)
        if key == ord('q'):
            break

    #out.release()
    cv2.destroyAllWindows()
except:
    TelegramBot.sendMessage('421681229', 'error ocured point={} chanel={}'.format(point, str(channel)))
