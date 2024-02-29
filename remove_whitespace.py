import os
import numpy as np
import matplotlib.pyplot as plt
import math

filtered_list = np.array([])
signal_count = 100*2*256*256 # We want 100 frames per time
position = 0
directory = 'FaradayCageTesting/bluetooth-cage-vs-no-cage.fc32'
signal_size = os.path.getsize(directory)
sample_count = int(np.floor(signal_size/(8*signal_count)))

max_MB = 500
file_count = 1

for i in range(sample_count-1):
	Data = np.fromfile('FaradayCageTesting/bluetooth-cage-vs-no-cage.fc32', dtype=np.single, count=signal_count, offset=signal_count*8*i)
	Real = Data[0::2]
	Imag = Data[1::2]
	Complex = Real + 1j*Imag

	window = 64
	data_array_partial = np.reshape(Complex, [-1, int(window/2)])

	average = np.mean(np.abs(data_array_partial),axis=1)
	Threshold = 2*np.mean(average) # This is only for bluetooth
	mask = average > Threshold

	buffer_val = 10
	change = np.append(np.logical_xor(mask[0:-2], mask[1:-1]), [False])

	for x in range (-buffer_val, buffer_val+1):
	    mask = np.logical_or(mask, np.roll(mask,x))
	    
	signal_array = data_array_partial[mask, :]

	filtered_list = np.concatenate((filtered_list, signal_array.reshape(-1,)))
		
	if len(filtered_list) > 4e6*max_MB:
		real_filtered = np.real(filtered_list)
		imag_filtered = np.imag(filtered_list)
		
		
		complex_filtered = np.empty((real_filtered.size+imag_filtered.size))
		complex_filtered[0::2] = real_filtered
		complex_filtered[1::2] = imag_filtered
		# This is where we generate the fc32 file
		complex_filtered.astype('float32').tofile("bluetooth/bluetooth_no_whitespace_"+f"{file_count:04d}"+".fc32")
		print("Created new file bluetooth_no_whitespace_"+f"{file_count:04d}"+".fc32")
		file_count += 1
		filtered_list = np.empty(0,dtype=np.single)
		

real_filtered = np.real(filtered_list)
imag_filtered = np.imag(filtered_list)
complex_filtered = np.empty((real_filtered.size+imag_filtered.size))
complex_filtered[0::2] = real_filtered
complex_filtered[1::2] = imag_filtered
complex_filtered.astype('float32').tofile("bluetooth/bluetooth_no_whitespace_"+f"{file_count:04d}"+".fc32") 
print("Created new file bluetooth_no_whitespace_"+f"{file_count:04d}"+".fc32")
