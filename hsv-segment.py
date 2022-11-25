import cv2
import numpy as np 
import streamlit as st
from PIL import Image
from PIL import ImageColor
from io import BytesIO
import base64
from ads import css_string

def rgb2hsv(r, g, b):
	# Normalize R, G, B values
	r, g, b = r / 255.0, g / 255.0, b / 255.0
 
	# h, s, v = hue, saturation, value
	max_rgb = max(r, g, b)    
	min_rgb = min(r, g, b)   
	difference = max_rgb - min_rgb 
 
	# if max_rgb and max_rgb are equal then h = 0
	if max_rgb == min_rgb:
    		h = 0
	 
	# if max_rgb==r then h is computed as follows
	elif max_rgb == r:
    		h = (60 * ((g - b) / difference) + 360) % 360
 
	# if max_rgb==g then compute h as follows
	elif max_rgb == g:
    		h = (60 * ((b - r) / difference) + 120) % 360
 
	# if max_rgb=b then compute h
	elif max_rgb == b:
    		h = (60 * ((r - g) / difference) + 240) % 360
 
	# if max_rgb==zero then s=0
	if max_rgb == 0:
    		s = 0
	else:
    		s = (difference / max_rgb) * 100
 
	# compute v
	v = max_rgb * 100
	# return rounded values of H, S and V
	return tuple(map(round, (h, s, v)))


def segment_color(img, lb=[0,10,255], ub=[30, 255, 255]):
	hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
	color_mask = cv2.inRange(hsv_img, np.array(lb), np.array(ub))
	return color_mask


def write_color(color_val, cols):
	rgb = ImageColor.getcolor(color_val, "RGB")
	r,g,b = rgb[0], rgb[1], rgb[2]
	hsv = rgb2hsv(r, g, b)
	cols[0].write(f"RGB : {rgb}")
	cols[0].write(f"HSV : {hsv}")
	return hsv


# Create application title and file uploader widget.
st.title("Color Segmentation using OpenCV")
st.sidebar.title('Select Image')

buffer = st.sidebar.file_uploader(' ', type=['jpg', 'jpeg', 'png'])

st.sidebar.header('STEPS:')
st.sidebar.text("1️⃣ Upload image")
st.sidebar.text("2️⃣ Select color")
st.sidebar.text("3️⃣ Adjust sliders")
st.sidebar.text(" ")

st.sidebar.markdown(css_string, unsafe_allow_html=True)
cols = st.columns(4)

color_val = cols[3].color_picker('Pick a Color', value='#2fefd1')

hsv = write_color(color_val, cols)

upper_bound = []
lower_bound = []

if buffer is not None:
	raw_bytes = np.asarray(bytearray(buffer.read()), dtype=np.uint8)
	# Loads image in a BGR channel order.
	image = cv2.imdecode(raw_bytes, cv2.IMREAD_COLOR)

	# Create placeholders to display input and output images.
	placeholders = st.columns(2)
	
	# Display Input image in the first placeholder.
	placeholders[0].image(image, channels='BGR')
	placeholders[0].text("Input Image")

	hue = st.slider('Hue', 0, 255, [hsv[0], hsv[0]+40])
	sat = st.slider('Saturation', 0, 255, [hsv[1], 255])
	val = st.slider('Lightness', 0, 255, [hsv[2], 255])

	lower_bound = [hue[0], sat[0], val[0]]
	upper_bound = [hue[1], sat[1], val[1]]

	if len(lower_bound) > 2:
		with cols[1]:
			st.write('HSV Lower Bound: ')
			st.write('HSV Upper Bound: ')
		with cols[2]:
			st.write(str(lower_bound))
			st.write(str(upper_bound))

	mask = segment_color(image, lb=lower_bound, ub=upper_bound)

	# Display mask.
	placeholders[1].image(mask)
	placeholders[1].text('Color Mask')

hide_streamlit_style = """
            <style>
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True) 