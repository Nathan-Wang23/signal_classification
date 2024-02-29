import numpy as np
import math
import os
import matplotlib.pyplot as plt
from PIL import Image
import glob
from random import randrange
# import datatorch

def main():

	nfs = "/nfs/nas"
	game_dict = {
		0: ["20191102_Pittsburg", "pitt19"], 
		1: ["20160910_mercer", "merc16"], 
		2: ["20160917_vanderbilt", "vand16"], 
		3: ["20160922_clemson", "clem16"], 
		4: ["20161001_miami", "miam16"], 
		5: ["20161015_ga_southern", "gaso16"], 
		6: ["20161029_duke", "duke16"], 
		7: ["20161119_virginia", "virg16"]
	}
	# Example of calling random game
	game_number = randrange(len(game_dict))
	rfsn = 2
	epoch = 12
	game = game_dict[1]
	game_directory = f"{nfs}/{game[0]}/rfsn{rfsn:01d}/pred/epoch{epoch:d}.sc16"
	game_name = f"{game[1]}_n{rfsn:01d}_e{epoch:02d}"
	number_of_frames = 25
	game_start = 200
	generate_frames(game_directory,game_name,game_start,number_of_frames)

	bt_dict = {
		0: ["bluetooth/no_whitespace/bluetooth-cage-vs-no-cage_0001.fc32","bt00"]
	}

	drone_dict = {
		0: ["drones/drone-mavic-air-2_00.cf32", "drone00"], 
		1: ["drones/drone-mavic-air-2-cage-vs-no-cage.cf32", "drone01"], 
		2: ["drones/drone-mavic-air-temp.cf32", "drone02"], 
		3: ["drones/drone-mavic-mini-cage-vs-no-cage.cf32", "drone03"], 
		4: ["drones/mavic_mini_controller_no_video.fc32", "drone04"], 
		5: ["drones/mavic_mini_controller_sequence_no_video.fc32", "drone05"], 
		6: ["drones/mavic_mini_controller_video.fc32", "drone06"], 
		7: ["drones/mavic_mini_drone_no_video.fc32", "drone07"], 
		8: ["drones/mavic_mini_drone_video.fc32", "drone08"]
	}

	wifi_dict = {
		0: ["wifi/SSH_gnuradio_test_March_22.fc32" , "wifi00"]
	}

	# Example of calling random drone test
	source_number = randrange(len(drone_dict))
	source = drone_dict[1]
	source_directory = f"{source[0]}"
	source_name = f"{source[1]}"
	number_of_frames = 25
	source_start = 100
	generate_frames(source_directory,source_name,source_start,number_of_frames)

	# Example of combining the 2
	ratio = 2.1
	number_of_frames = 25
	generate_frames(source_directory,source_name,source_start,number_of_frames,nd=game_directory,nn=game_name,ns=game_start,r=ratio)

	# file_size = os.path.getsize(directory)
	# sample_count = int(np.floor(file_size/(data_size*frames_count/4))) # The total number of 256*256 frames we have
	

def generate_frames(directory,base_name,start,number_of_frames,**noise_data):

	# api = datatorch.api.ApiClient(api_key = '48ce0010-3f20-4c4e-9ec2-4b00e756e0b0')
	# proj = api.project('6556a98f-eb54-4766-b770-428fc481f531')

	cmap = plt.get_cmap('viridis') # Set the colormap for creating images
	
	if ('nd' in noise_data) and ('nn' in noise_data):
		add_noise = True
		noise_directory = noise_data.get('nd')
		noise_name = noise_data.get('nn')
		folder = base_name + '_' + noise_name
		if ('ns' in noise_data):
			noise_start = noise_data.get('ns')
		else:
			noise_start = 0
		if ('r' in noise_data):
			ratio = noise_data.get('r')
		else:
			ratio = 1
	else:
		add_noise = False
		folder = base_name
		

	# Check whether the specified path exists or not
	isExist = os.path.exists("Images/" + folder)

	if not isExist:
		# Create a new directory because it does not exist 
		os.makedirs("Images/" + folder)	 
	
	max_frames_at_once = 20

	reps = math.ceil(number_of_frames/max_frames_at_once)

	for i in range(reps):
		Complex = grab_data(directory, base_name, start, min(number_of_frames, max_frames_at_once))
		
		if add_noise:
			Noise= grab_data(noise_directory, noise_name, noise_start, min(number_of_frames, max_frames_at_once))
			P_noise = abs(np.sum(np.multiply(Noise,Noise.conjugate())))
			P_source = abs(np.sum(np.multiply(Complex,Complex.conjugate())))
			scale_factor = math.sqrt(ratio*P_noise/P_source) # Energy calculted as sum of square magnitudes, so we need the square root of the scaling factor
			Complex = np.add(np.multiply(Complex,scale_factor), Noise)

		window = 64
		data_array_partial = np.reshape(Complex, [-1, int(window/2)])

		signal_overlap_array = np.append(data_array_partial[:-1,:], data_array_partial[1:,:],1)

		nfft = 256
		Spectral_data = np.transpose(np.fft.fft(signal_overlap_array, nfft, 1))
		Spectral_data = np.fft.fftshift(Spectral_data, axes=(0,))
		Spectral_data = np.flip(Spectral_data, axis=0)

		Sxx = np.abs(Spectral_data)

		# Sxx is the absolute value spectrogram array we are interested in
		# The input to our ML algorithm shouldn't be the entire array, but instead chunks of it like this
		width = 256
		step = 128
		nsteps = min(number_of_frames, max_frames_at_once)
		
		Sxx_array =  [Sxx[:, i*step:i*step + width] for i in range(0, nsteps)]
		
		for k in range(len(Sxx_array)):
			Sxx = Sxx_array[k]
			
			Szz = preprocess_frame(Sxx)

			# Save the frame as an image
			# Apply the colormap like a function to any array:
			colored_image = cmap(Szz)

			# Obtain a 4-channel image (R,G,B,A) in float [0, 1]
			# But we want to convert to RGB in uint8 and save it.
			# name = path + "/img/" + source + f"_{image_count:04d}.png"
			if add_noise:
				name = f"Images/{folder}/{base_name}_{start+k:04d}_{noise_name}_{noise_start+k:04d}_r{int(ratio*100):03d}.png"
			else:
				name = f"Images/{folder}/{base_name}_{start+k:04d}.png"
			Image.fromarray((colored_image[:, :, :3] * 255).astype(np.uint8)).save(name)

			#testfile = open(name, 'rb')
			#api.upload_to_default_filesource(proj,testfile)

		if add_noise:
			noise_start += max_frames_at_once

		number_of_frames -= max_frames_at_once
		start += max_frames_at_once
		
	print("Added images to: Images/" + folder)
	return

def grab_data(directory, base_name, start, number_of_frames):

	data_type = directory.partition(".")[-1]
	if "16" in data_type:
		data_size = 16
	elif "32" in data_type:
		data_size = 32
	else:
		print("Error: Data type not valid. Expected fc32 or sc16.")			
		return

	step = 128
	window = 64 # complex data so 64 samples gives only 32 complex values
	frame_count = (number_of_frames + 1)*window*step + window # We want 50 frames per time
	offset_size = start*window*step
		
	if data_size == 16:
		Data = np.fromfile(directory, dtype=np.int16, count=frame_count, offset=offset_size*2)
		Data = np.float64(Data)
	elif data_size == 32:
		Data = np.fromfile(directory, dtype=np.single, count=frame_count, offset=offset_size*4)
		Data = np.float64(Data)

	# print("Successfully read in fc32 file")
	if np.isnan(Data).any():
		print("Data Contains NaN")
		return

	Real = Data[0::2]
	Imag = Data[1::2]
	Complex = Real + 1j *Imag
	return Complex

def preprocess_frame(Sxx):
	# My goal here was to compress the values to the interval [0,1] with a mean of .5
	# Subtract the minimum value to set the new minimum to 0
	Sxx_0 = Sxx - np.amin(Sxx)
	
	# We want to raise all walues to a power of n such that the average goes to half the maximum
	# (a/b)^n = 1/2 is equivalent to solving the base (a/b) log of (1/2)
	# where a is the average and b is the maximum
	avg = np.mean(Sxx_0)
	maximum = np.amax(Sxx_0)
	exponent = math.log(.5, avg/maximum)
	
	# Raising every value to the power of n (which is a fraction) and then dividing by the maximum will get us close
	Syy = np.float_power(Sxx_0,exponent)
	
	# It doesn't work exactly because E[x^n] is not equal to E[x]^n, but it will iteratively approach the correct value
	# here doing it once results in an average of 0.4325 and 2 passes results in an average of 0.4989 (after dividing by the maximum value)
	avg = np.mean(Syy)
	maximum = np.amax(Syy)
	exponent = math.log(.5, avg/maximum)
	Szz = np.float_power(Syy,exponent)
	
	# Diziding by the maximum restricts all values to the interval [0,1]
	Szz = Szz/np.amax(Szz)

	return Szz


if __name__ == "__main__":
    main()
