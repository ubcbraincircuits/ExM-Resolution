# !pip install javabridge
# !pip install python-bioformats

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline
from scipy.signal import argrelextrema
import cv2

import javabridge
import bioformats
from radial_data import radial_data

javabridge.start_vm(class_path=bioformats.JARS)

file_path = "Test.czi" # Edit this for whichever .czi file you are dealing with
radial_file = "radial_data.py"

# Get info about the channels and pixel size from the metadata
omexmlstr = bioformats.get_omexml_metadata(path=file_path)
o = bioformats.OMEXML(omexmlstr)
pixels=o.image().Pixels

num_channels = pixels.get_channel_count()

physical_size_x = pixels.get_PhysicalSizeX()
x_units = pixels.get_PhysicalSizeXUnit()
x_pix = pixels.get_SizeX()
physical_size_y = pixels.get_PhysicalSizeY()
y_units = pixels.get_PhysicalSizeYUnit()
y_pix = pixels.get_SizeY()

# Variables to help read the image and take its fft
chan_num = 2 # any number between 0 and num_channels - 1
fft_size = int(x_pix/2)

image = bioformats.load_image(file_path,c=chan_num)*65535
f = np.fft.fft2(image)
fshift = np.fft.fftshift(f)
magnitude_spectrum = np.log(np.abs(fshift))*2

# Show the image
fig,ax = plt.subplots(1)
plt.imshow(image, cmap = 'gray')

# Show the magnitude spectrum
fig,ax = plt.subplots(1)
plt.imshow(magnitude_spectrum, cmap = 'gray')
plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])

wav_color_dict = {200: 'purple',
                  300: 'blue',
                  400: 'yellow',
                  500: 'red'}

for wavelength in range(200,600,100):
  wav_pix = physical_size_x/wavelength*1000
  print(wav_pix)
  radius = wav_pix*804
  circ = plt.Circle((402,402),radius,fill=False,label=str(wavelength) + " nm",
                    color=wav_color_dict[wavelength])
  ax.add_patch(circ)

plt.legend(prop={'size': 6})

# Analyze the magnitude spectrum
data = radial_data(magnitude_spectrum, annulus_width=2, rmax=fft_size)
x = (data.r/x_pix)**2
y = data.mean

# Manually set left and right thresholds for now
left_threshold = 0.02
right_threshold = 0.1

x_lin = x[np.where(x > left_threshold)[0]]
x_lin = x_lin[np.where(x_lin < right_threshold)[0]]

y_lin = y[np.where(x > left_threshold)[0]]
y_lin = y_lin[np.where(x_lin < right_threshold)[0]]

model = np.polyfit(x_lin, y_lin, 1)
slope = model[0]
intercept = model[1]

y_lin = slope * x_lin + intercept
fig = plt.figure()
plt.plot(x,y,'ro',label = 'data',markersize=2)
plt.plot(x_lin,y_lin, c='b')

xstdev = np.sqrt(np.abs(slope))/(2*np.pi)
fwhm = xstdev * 2 * np.sqrt(2*np.log(2))

# print("Slope: " + str(slope))
# print("Standard deviation: " + str(xstdev))
print("Resolution in pixels: " + str(fwhm))
print("Resolution in micrometers: " + str(fwhm*physical_size_x))
