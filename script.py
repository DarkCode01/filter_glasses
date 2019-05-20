import cv2
import sys


def main() -> None:
    make(sys.argv[1])

def make(img: 'This is the path of image face') -> object:
    img = cv2.imread(img)

    cv2.imshow('input', img)

    glasses = cv2.imread('./imgs/glasses.png', -1)
    glasses = cv2.resize(glasses, (0, 0), fx=0.5, fy=0.5)

    face = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    eyes = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        face_gray = gray[y:y + h, x:x + w]
        face_color = img[y:y + h, x:x + w]

        _eyes = eyes.detectMultiScale(face_gray)
        x_offset = 0
        y_offset = 0

        for (ex, ey, ew, eh) in _eyes:
            x_offset = x_offset + ex
            y_offest = y_offset + ey

        x = (ex // 4) + 40
        y = (ey // 4) - 20

        y1, y2 = y, y + glasses.shape[0]
        x1, x2 = x, x + glasses.shape[1]

        alpha_s = glasses[:, :, 3] / 255.0
        alpha_l = 1.0 - alpha_s


        for c in range(0, 3):
            face_color[y1:y2, x1:x2, c] = (alpha_s * glasses[:, :, c] + alpha_l * face_color[y1:y2, x1:x2, c])

    cv2.imshow('output', img)

if __name__ == '__main__':
    main()
    cv2.waitKey(0)
