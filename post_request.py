import requests
import io
from PIL import Image
import zlib

response = requests.post('http://localhost:5000/detect_objects', 
                         files={'file': open('E:\Code\Task03\images4.jpeg', 'rb')})
binary_data = response.content

decompressed_data = zlib.decompress(binary_data)

img = Image.open(io.BytesIO(decompressed_data))
img.save('E:/Code/Task03/output_image.png')
