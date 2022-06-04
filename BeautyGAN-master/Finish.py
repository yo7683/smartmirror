import pygame
import ctypes
import speech_recognition as sr
import dlib
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import pymysql
import os
import random
from selenium import webdriver
import time
import pyautogui
import cv2
# 64bit, python3.6.8, tensorflow 1.9 or 1.5.1, pyaudio
# dlib 19.23.1 , speechrecognition 3.8.1
# matplotlib 3.3.4 , numpy 1.19.5 , cmake
#pyscreeze locateOnWindow'

# 음성인식  
r = sr.Recognizer()
mic = sr.Microphone()
quit = ['그만','취소','꺼 줘','닫아 줘','종료']
text = ''
def callback(recognizer, audio):
    global text
    try:
        text = recognizer.recognize_google(audio,language = "ko_KR")
        print("You said " + text)
    except sr.UnknownValueError:
        print("Google Speech Recognition could not understand audio")
    except sr.RequestError as e:
        print("Could not request results from Google Speech Recognition service; {0}".format(e))
with mic as source:
    r.adjust_for_ambient_noise(source)
#stop_listening = r.listen_in_background(mic, callback,phrase_time_limit=5)
stop_listening = r.listen_in_background(mic, callback)


# 데이터베이스
conn = pymysql.connect(host='132.226.171.153',user='trio',password='qwer1234',db='SMARTMIRROR',charset='utf8')
cursor = conn.cursor()

# 텐서플로 설정
sess = tf.Session()
sess.run(tf.global_variables_initializer())
saver = tf.train.import_meta_graph('models/model.meta')
saver.restore(sess, tf.train.latest_checkpoint("models"))
graph = tf.get_default_graph()
X = graph.get_tensor_by_name('X:0') #source
Y = graph.get_tensor_by_name('Y:0') #reference
Xs = graph.get_tensor_by_name('generator/xs:0') #output

# 컬러 딕셔너리
color_dict = {
    "Red"  : ["Red","red","RED","빨강","빨간","빨간색","붉은","붉은색","레드"],
    "Orange" : ["Orange","oeange","ORANGE","오렌지","오랜지","오랜지색","오렌지색","주황","주황색"],
    "Yellow" : ["Yellow","yellow","YELLOW","노랑","노란","노란색","옐로우"],
    "Green" : ["Green","green","GREEN","초록","초록색","연두","연두색","그린"],
    "Blue" : ["Blue","blue","BLUE","파랑","파란","파란색","푸른","푸른색","남색","블루"],
    "Purple" : ["Purple","purple","PURPLE","보라","보라색","퍼플"],
    "White" : ["White","white","WHITE","흰","흰색","하얀","하얀색","밝은","화이트"],
    "Black" : ["Black","BLACK","black","검은","검정","검은색","검정색","어두운","블랙"],
    "Brown" : ["Brown", "BROWN", "brown","갈색","브라운","베이지"]
}

#Load Models
detector = dlib.get_frontal_face_detector()  #얼굴 영역 인식 모델 로드
sp = dlib.shape_predictor("models/shape_predictor_5_face_landmarks.dat")

# 얼굴의 각도를 똑바로 맞춰주는 작업
def align_faces(img):
    dets = detector(img,1)
    objs = dlib.full_object_detections()
    for detection in dets:
        s = sp(img, detection)
        objs.append(s)
    # padding을 작게 하면 얼굴을 더 작은 범위로 지정한다
    faces = dlib.get_face_chips(img, objs, size=256, padding=0.35)
    return faces 

# 전처리 작업
def preprocess(img):
    return img.astype(np.float32) / 127.5 -1           #0 ~ 255 -> -1 ~ 1
# 후처리 작업
def postprocess(img):
    return ((img+1.)*127.5).astype(np.uint8)           #-1 ~ 1 -> 0 ~ 25

# 유튜브 화면 띄우기
def __youtube(id):
    global driver
    # 크로미움 설정
    options = webdriver.ChromeOptions()
    options.add_experimental_option("excludeSwitches",["enable-logging"])
    options.add_extension('ublock.crx')
    driver = webdriver.Chrome('./chromedriver.exe',options=options)
    driver.set_window_position(0,0)
    driver.maximize_window()
    driver.get("https://www.youtube.com/watch?v="+id)
    time.sleep(2)
    
    # 팝업창 닫기 2번
    pyautogui.press("esc")
    time.sleep(0.1)
    pyautogui.press("esc")
    time.sleep(0.1)
    # 전체화면
    try:
        #영상 재생 버튼
        driver.find_element_by_xpath('//*[@id="movie_player"]/div[4]/button').click()
    except:
        pass
    time.sleep(0.1)
    try:
        #전체화면
        pyautogui.press('f')
    except:
        pass

# db에서 자료 가져오기
def __sql(sql_text):
    cloth = 0
    hair = 0
    for i in range(len(sql_text)):
        if "옷" in sql_text[i]:
            cloth = sql_text[i-1]
        # 형용사, 명사에 따라 띄어쓰기가 달라짐
        if "머리" in sql_text[i]:
            index = sql_text[i].find("머리")
            if index != 0:
                hair = sql_text[i][:index]
            else:
                hair = sql_text[i-1]
    # sql에 넣을 값 가져오기
    for key, value in color_dict.items():
        if cloth in value:
            cloth = key
        if hair in value:
            hair = key
    # sql 작성
    if cloth != 0 and hair != 0:
        sql = "SELECT id, url FROM YOUTUBE WHERE cloth "+"= '"+cloth+"' and hair "+"= '"+hair+"'"
    elif cloth == 0 and hair != 0:
        sql = "SELECT id, url FROM YOUTUBE WHERE hair "+"= '"+hair+"'"
    elif cloth != 0 and hair == 0:
        sql = "SELECT id, url FROM YOUTUBE WHERE cloth "+"= '"+cloth+"'"
    else:
        sql = "SELECT id, url FROM YOUTUBE"
    print(sql)
    cursor.execute(sql)
    result = cursor.fetchall()
    return result

# 합성 프로세스
def process(result):
    makeup_imgs = []
    img2_faces = []
    ref_imgs = []
    output_imgs =[]
    pygame_img = []
    global ids, text
    
    text_start = font.render("사진을 찍으려면 '김치' 또는 '스마일'을 외쳐주세요",True,(255,255,255))
    background = pygame.Surface(screen.get_size())
    rect = text_start.get_rect()
    rect.centerx = u32.GetSystemMetrics(0) / 2
    rect.centery = u32.GetSystemMetrics(1) / 5
    background.blit(text_start,rect)
    screen.blit(background,(0,0))
    pygame.display.flip()
    
    #camera
    cap = cv2.VideoCapture(1) # or (1)
    cap.set(3,640)
    cap.set(4,480)
    cv2.namedWindow('camera')
    cv2.moveWindow('camera',int(u32.GetSystemMetrics(0)/2 - 320), int(u32.GetSystemMetrics(1)/2 - 240))
    while True:
        ret, frame = cap.read()
        cv2.flip(frame,1)
        if ret:
            cv2.imshow('camera',frame)
            if cv2.waitKey(1):
                pass
            if text == '김치' or text == '스마일':
                cv2.imwrite('imgs/self.jpg',frame)
                break
        pass
    cap.release()
    cv2.destroyAllWindows()
    
    # 맨얼굴 가져오기
    no_makeup = dlib.load_rgb_image("imgs/self.jpg")
    img1_faces = align_faces(no_makeup)
    
    for record in result:
        id = record[0]
        ids.append(id)
        url = record[1]
        os.system("curl "+ url  + ' > D:/BeautyGAN-master/imgs/imgur/' + id + '.jpg')
        tmp = dlib.load_rgb_image("imgs/imgur/"+id+'.jpg')
        makeup_imgs.append(tmp)
    # 이미지 align
    for index in makeup_imgs:    
        img2_faces.append(align_faces(index))
    
    # 소스 이미지 (no_makeup)
    src_img = img1_faces[0]
    # 레퍼런스 이미지 makeup
    for face in img2_faces:
        ref_imgs.append(face[0])
    
    # 사진 합성
    X_img = preprocess(src_img)
    X_img = np.expand_dims(X_img,axis=0) #np.expand_dims() : 배열에 차원을 추가한다. 즉, (256,256,2) -> (1,256,256,3)
    for ref_img in ref_imgs:
        Y_img = preprocess(ref_img)
        Y_img = np.expand_dims(Y_img,axis=0) #텐서플로에서 0번 axis는 배치 방향
        output = sess.run(Xs, feed_dict={
        X: X_img,
        Y: Y_img
        })
        output_imgs.append(postprocess(output[0]))
    
    # 합성 사진 저장 후 pygame으로 불러오기
    for output_img in output_imgs:
        plt.imsave('imgs/test.jpg',output_img)
        pyimg = pygame.image.load("imgs/test.jpg")
        pyimg = pygame.transform.scale(pyimg,(350,350))
        pygame_img.append(pyimg)
        
    # 사진을 백그라운드에 넣기
    background = pygame.Surface(screen.get_size())
    text_start = font.render("원하시는 사진의 번호를 골라주세요.",True,(255,255,255))
    rect = text_start.get_rect()
    rect.centerx = u32.GetSystemMetrics(0) / 2
    rect.centery = u32.GetSystemMetrics(1) / 4
    background.blit(text_start,rect)
    
    cnt = len(pygame_img)
    for i in range(len(pygame_img)):
        rect = pygame_img[i].get_rect()
        rect.centerx = u32.GetSystemMetrics(0) / (2+cnt-1) * (i+1)
        rect.centery = u32.GetSystemMetrics(1) / 2
        background.blit(pygame_img[i],rect)
    time.sleep(0.1)
    screen.blit(background,(0,0))
    pygame.display.flip()
    
########
# main #
########
if __name__ == '__main__':
    global u32, screen
    #pygame
    u32 = ctypes.windll.user32
    pygame.init()
    screen = pygame.display.set_mode((u32.GetSystemMetrics(0), u32.GetSystemMetrics(1)))
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("malgungothic",30)
    clock.tick(10)
    
    # boolean
    yt = False
    wait = True
    choice_token = False
    command_token = False
    done = True
    # loop
    while done:
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = False
                
        # 대기화면
        if wait:
            sql_result = '' 
            ids = []
            text_start = font.render("안녕하세요.",True,(255,255,255))
            rect = text_start.get_rect()
            rect.centerx = u32.GetSystemMetrics(0) / 2
            rect.centery = u32.GetSystemMetrics(1) / 2
            background = pygame.Surface(screen.get_size())
            background.blit(text_start,rect)
            screen.blit(background,(0,0))
            pygame.display.flip()
        
        # 스마트미러 인식 대기
        if command_token == False and choice_token == False:
            if text == "스마트미러" or text == "스마트 미러":
                text_start = font.render("네 말씀하세요.",True,(255,255,255))
                background = pygame.Surface(screen.get_size())
                background.blit(text_start,rect)
                screen.blit(background,(0,0))
                pygame.display.flip()
                wait = False
                command_token = True
                
        # 프로세스 동작
        if command_token == True and choice_token == False and yt == False:
            if "화장" in text:
                # sql 작성
                sql_result = __sql(sql_text = text.split())
                # sql_result가 복수의 데이터면
                try:
                    if len(sql_result) > 1:
                        if len(sql_result) > 2:
                            sql_result = random.sample(sql_result,3)
                        else:
                            sql_result = random.sample(sql_result,2)
                        process(sql_result)
                        choice_token = True
                    elif len(sql_result) == 1:
                        process(sql_result)
                        choice_token = True
                    else:
                        text_start = font.render("자료가 없습니다.",True,(255,255,255))
                        background = pygame.Surface(screen.get_size())
                        background.blit(text_start,rect)
                        screen.blit(background,(0,0))
                        pygame.display.flip()
                        time.sleep(5)
                        command_token = False
                        wait = True
                        text = ''
                        choice_token = False
                except:
                    text_start = font.render("합성에 실패하였습니다.",True,(255,255,255))
                    background = pygame.Surface(screen.get_size())
                    background.blit(text_start,rect)
                    screen.blit(background,(0,0))
                    pygame.display.flip()
                    time.sleep(5)
                    command_token = False
                    wait = True
                    text = ''
                    choice_token = False
            elif text == "취소":
                command_token = False
                text = ''
                wait = True
                choice_token = False
                pass
        
        # 원하는 사진 고르기
        if choice_token == True and yt == False:
            if text == "1번" or text == "일번" or text == "첫 번째" or text =="일본" or text == "틀어" or text == "틀어 줘" or text == "보여 줘":
                print(ids[0])
                __youtube(ids[0])
                yt = True
            elif text == "2번" or text == "이번" or text == "두 번째":
                print(ids[1])
                __youtube(ids[1])
                yt = True
            elif text == "3번" or text == "삼번" or text == "세 번째":
                print(ids[2])
                __youtube(ids[2])
                yt = True
            elif text == "취소":
                command_token = False
                text = ''
                wait = True
                choice_token = False
                print("취소")
                
        # 영상시청 종료
        if yt:
            if text in quit:
                driver.close()
                command_token = False
                text = ''
                wait = True
                choice_token = False
                yt = False
                time.sleep(1)
                
        if text == '스마트미러 종료' or text == '스마트 미러 종료':
            break
        
    pygame.quit()