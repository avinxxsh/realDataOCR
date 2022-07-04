import cv2 as cv
import numpy as np
from pdf2image import convert_from_path
import PyPDF2
import pytesseract
from PIL import Image
import re

file = open('pdf_name.pdf', 'rb')
readpdf = PyPDF2.PdfFileReader(file)
count = readpdf.numPages
print('Page Count ==>', count)

a = 0
open('paths.txt', 'w').close()
while a < count:
    file1 = open('paths.txt', 'a')
    file1.write("C:\\Users\\dgavi\\PycharmProjects\\realDataOCR\\image" + str(a) + ".jpg\n")
    a += 1

images = convert_from_path('pdf_name.pdf')

for i, image in enumerate(images):
    fname = "image" + str(i) + ".jpg"
    image.save(fname, 'JPEG')

img = np.array(Image.open('image0.jpg'))
# cv.imshow("Converted Image", img)


def grayscale(image):
    return cv.cvtColor(image, cv.COLOR_BGR2GRAY)


gray_image = grayscale(img)
cv.imwrite("temp/grey.jpeg", gray_image)
# cv.imshow("Gray Image", gray_image)
cv.waitKey(0)

# converting image to grayscale is only the 1st step of binarization - makes the process easier
thresh, im_bw = cv.threshold(gray_image, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C + cv.THRESH_BINARY, 11, 2)
# thresh, im_bw = cv.threshold(gray_image, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
# the integers entered are pixel values where 0 is black, 127 is the mid-tone point and 255 is white
cv.imwrite("temp/bw_image.jpeg", im_bw)
# cv.imshow("B/W Image", im_bw)


pytesseract.pytesseract.tesseract_cmd = "C:\\Program Files\\Tesseract-OCR\\tesseract.exe"
img_1 = cv.cvtColor(im_bw, cv.COLOR_BGR2RGB)
details = pytesseract.image_to_string('paths.txt')
# details = pytesseract.image_to_string("paths.txt")
print(details)
# print(pytesseract.image_to_boxes(img_1))

# showing character detection
# hImg, wImg = img_1.shape[0:2]
# boxes = pytesseract.image_to_boxes(no_noise)
# for b in boxes.splitlines():
#     b = b.split(' ')
#     # print(b)
#     x, y, w, h = int(b[1]), int(b[2]), int(b[3]), int(b[4])
#     cv.rectangle(img_1, (x, hImg - y), (w, hImg - h), (0, 0, 255), 2)

# detecting words
hImg, wImg = img_1.shape[0:2]
boxes = pytesseract.image_to_data(img_1)
for x, b in enumerate(boxes.splitlines()):
    if x != 0:
        b = b.split()
        print(b)
        if len(b) == 12:
            x, y, w, h = int(b[6]), int(b[7]), int(b[8]), int(b[9])
            cv.rectangle(img_1, (x, y), (w + x, h + y), (0, 0, 225), 3)
            cv.putText(img_1, b[11], (x, y), cv.FONT_HERSHEY_PLAIN, 1, (50, 50, 255), 2)


cv.imshow('Final Image', img_1)

with open('result_text.txt', 'w') as file:
    for detail in details:
        file.write(detail)

# regex
print("\nRegEx Batch No. Match Results -->")
# pattern = re.compile(r"\b(?=[^\W\d_]*\d)(?=\d*[^\W\d_])[^\W_]{6,9}\b")
# pattern = '^[a-zA-Z0-9]{6,9}$'
# pattern = re.compile('^\w+$')
pattern = re.compile(r'\b(?=[a-zA-Z0-]*\d)[A-Za-z0-9-]{5,11}\b') # best results so far
for i, line in enumerate(open('result_text.txt')):
    for match in re.finditer(pattern, line):
        print(match.group())

cv.waitKey(0)
cv.destroyAllWindows()
# r'\w+(?:-\w+)+'
