import cv2
import mediapipe as mp
import requests
import base64
import torchvision.models as models
import torchvision
import torch
import numpy as np
import torch.nn as nn
import torchvision.transforms as transforms

mpPose = mp.solutions.pose
pose = mpPose.Pose()
mpDraw = mp.solutions.drawing_utils
vid = cv2.VideoCapture(0)
while True:
    # Capture the video frame
    # by frame
    ret, frame = vid.read()

    # Display the resulting frame
    imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    trans = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize([512, 512]),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
    img = trans(imgRGB).unsqueeze(0)
    print(img.size())

#------- Code for classification inference part ----------
    model = models.resnet18(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 10)
    model.load_state_dict(torch.load('/Users/wuxindi/Downloads/meterbeater_dataset/epoch_41_ckpt.pt', map_location='cpu'))
    
    outputs = model(img)
    print(outputs)
    # Let us grab the top5 probability indices
    idx_asc = outputs.argsort()
    idx_desc = np.fliplr(idx_asc)
    idx_top5 = idx_desc[0, :10]
    print(idx_desc)
    
    # print(idx_top5)
    r = 1
    # Now for each of the index create a text and put on displayed frame
    classes = ["cop","person_1","person_2","person_3","person_4","person_5","person_6","person_7","person_8","person_9"]
    for id in idx_desc[0]:
        text = "{}: probability {:.3} %".format(classes[id], outputs[0, id] * 100)
        cv2.putText(frame, text, (0, 25 + 40 * r), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        r += 1


    # the 'q' button is set as the
    # quitting button you may use any
    # desired button of your choice
    results = pose.process(imgRGB)
    min = [200,50]
    if results.pose_landmarks:
        mpDraw.draw_landmarks(frame, results.pose_landmarks,mpPose.POSE_CONNECTIONS)
        ls = results.pose_landmarks.landmark[11]
        rs = results.pose_landmarks.landmark[12]
        lw = results.pose_landmarks.landmark[15]
        rw =  results.pose_landmarks.landmark[16]
        h, w, c = frame.shape
        lsx, lsy = int(ls.x * w), int(ls.y * h)
        rsx, rsy = int(rs.x * w), int(rs.y * h)
        lx, ly = int(lw.x * w), int(lw.y * h)
        rx, ry = int(rw.x * w), int(rw.y * h)
        
        if abs(lx) < lsx and abs(lx) > rsx and abs(rx) < lsx and abs(rx) > rsx and abs(ly - ry) < min[1]:
            if idx_desc[0][0] == 0:
               
                headers = {'Content-type': 'text/plain'}

                url = "https://m8qo5eeqz6.execute-api.us-east-1.amazonaws.com/payParking"
                image_file = 'frame.jpg'
                cv2.imwrite("frame.jpg", frame)
                with open(image_file, "rb") as f:
                    im_bytes = f.read()
                im_b64 = base64.b64encode(im_bytes).decode("utf8")
                # img = cv2.imread('frame.jpg')
                imageEncoding = im_b64
                # print(imageEncoding)
                payload = "{\r\n    \"username\": \"test\",\r\n    \"imageName\": \"test.jpeg\",\r\n    \"imageData\": \""+imageEncoding+"\"\r\n}"
                headers = {
                    'Content-Type': 'text/plain'
                }

                x = requests.request("POST", url, headers=headers, data=payload)
                if x.status_code == 200:
                    print("Parking Spot Purchased!")
                    break
                else:
                    print(x.reason)
                    print("failed")
                    break

        for id, lm in enumerate(results.pose_landmarks.landmark):
            cx, cy = int(lm.x * w), int(lm.y * h)
            cv2.circle(frame, (cx,cy), 5,(255,0,0),cv2.FILLED)



    cv2.imshow('frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # After the loop release the cap object
vid.release()
# Destroy all the windows
cv2.destroyAllWindows()

def get_base64_encoded_image(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode('utf-8')

#while True:
#    success, img = cap.read()
#    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#    results = pose.process(imgRGB)
#    print(results.pose_landmarks)
#    if results.pose_landmarks:
#        mpDraw.draw_landmarks(img, results.pose_landmarks,mpPose.POSE_CONNECTIONS)

#    cv2.imshow("Image", img)
#    cv2.waitKey(1)


