from PIL import Image
from pytesseract import image_to_string

filename = "somefile.png"
text = image_to_string(Image.open(filename))
print(text)