import sys
import cv2

def main(video: '''path of video''') -> object:
    video_cap = cv2.VideoCapture(video)
    loop(video_cap)
    cv2.destroyAllWindows()

def loop(video: ''' video ''') -> object:
    glasses = cv2.imread('./imgs/glasses.png', -1)
    glasses = cv2.resize(glasses, (125, 125))
    face = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    eyes = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

    while video.isOpened():
        ret, frame = video.read()
        video_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face.detectMultiScale(video_gray, 1.3, 5)

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            face_gray = video_gray[y:y + h, x:x + w]
            face_color = frame[y:y + h, x:x + w]

            _eyes = eyes.detectMultiScale(face_gray)
            x_offset = 0
            y_offset = 0

            for (ex, ey, ew, eh) in _eyes:
                x_offset = x_offset + ex
                y_offest = y_offset + ey

            x = (ex // 4)
            y = (ey // 4)

            y1, y2 = y, y + glasses.shape[0]
            x1, x2 = x, x + glasses.shape[1]

            alpha_s = glasses[:, :, 3] / 255.0
            alpha_l = 1.0 - alpha_s

            try:
                for c in range(0, 3):
                    face_color[y1:y2, x1:x2, c] = (alpha_s * glasses[:, :, c] + alpha_l * face_color[y1:y2, x1:x2, c])
            except Exception as e:
                pass

        cv2.imshow('video running', frame)

        if cv2.waitKey(1) and 0xFF == ord('q'):
            break

    video.release()

if __name__ == '__main__':
    print('asdasdasd')

    main(sys.argv[1])
