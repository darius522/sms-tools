# functions that implement analysis and synthesis of sounds using a modified version of the SineModel, 
# supporting a multi-resolution windowing approach

import numpy as np
from scipy.signal import blackmanharris, triang
from scipy.fftpack import ifft, fftshift
import math
import dftModel as DFT
import utilFunctions as UF

def sineModelMultiRes(x, fs, windows, fftSizes, t, bands):
	"""
	Analysis/synthesis of a sound using a multi-resolution sinusoidal model, without sine tracking
	x: input array sound, 
	s: sampling frequency, 
	w: analysis windows, 
	fftSizes: sizes of complex spectrum, 
	t: threshold in negative dB,
	bands: bands of the sounds for multi-resolution analysis

	returns y: output array sound
	"""

	# Resynthesis values
	Ns = 512                                                # FFT size for synthesis (even)
	H = Ns//4                                               # Hop size used for analysis and synthesis
	hNs = Ns//2                                             # half of synthesis FFT size
	yw = np.zeros(Ns)                                       # initialize output sound frame

	x = np.array(x)										    # Convert input to numpy array

	# Create output buffers for all the bands
	y1 = np.zeros(x.size)
	y2 = np.zeros(x.size)
	y3 = np.zeros(x.size)
	outputArrays = np.array([y1,y2,y3])

	sw = np.zeros(Ns)                                       # initialize synthesis window
	ow = triang(2*H)                                        # triangular window
	sw[hNs-H:hNs+H] = ow                                    # add triangular window
	bh = blackmanharris(Ns)                                 # blackmanharris window
	bh = bh / sum(bh)                                       # normalized blackmanharris window
	sw[hNs-H:hNs+H] = sw[hNs-H:hNs+H] / bh[hNs-H:hNs+H]     # normalized synthesis window

	# Compute analysis/synthesis 3 times with each window/fft/band
	for i in range(0, 3):

	#-----variable inits-----

		# Get the current window bounds (rounding up/floor down)
		hM1 = int(math.floor((windows[i].size+1)/2))
		hM2 = int(math.floor(windows[i].size/2))

		# Get start/end pin for current window
		pin = max(hNs, hM1)
		pend = x.size - max(hNs, hM1)

		# Normalize window
		w = windows[i] / sum(windows[i])

		# Get current FFT size and init FFT buffer
		N = fftSizes[i]
		fftbuffer = np.zeros(N)                

		# Pick up/down bin limits
		binCutoffUpLimit = (np.ceil(bands[i]*N/fs))-1
		if i == 0: 
			binCutoffDownLimit = 0 
		else: 
			binCutoffDownLimit = np.ceil(bands[i-1]*N/fs)

		while pin < pend:

		#-----analysis----- 

			# Get the frame with current window size            
			x1 = x[pin-hM1:pin+hM2]                          

			# Get the spectra for each frame sizes using windows with their respective FFT sizes
			mX, pX = DFT.dftAnal(x1, w, N)

			# Only get the part of the spectrum we're interested in for the current band
			mXFilt = mX.copy() 
			mXFilt[binCutoffDownLimit:] = -120
			mXFilt[:binCutoffUpLimit] = -120

			# Get the peaks out of each spectrum
			ploc = UF.peakDetection(mX, t)                      

			# Refine peak values by interpolation
			iploc, ipmag, ipphase = UF.peakInterp(mX, pX, ploc)  

			# Convert peak locations to Hertz 
			ipfreq = fs*iploc/float(N)                          

		#-----synthesis-----
			Y = UF.genSpecSines(ipfreq, ipmag, ipphase, Ns, fs)   # generate sines in the spectrum         
			fftbuffer = np.real(ifft(Y))                          # compute inverse FFT
			yw[:hNs-1] = fftbuffer[hNs+1:]                        # undo zero-phase window
			yw[hNs-1:] = fftbuffer[:hNs+1]
			# Place sample to respective output array
			outputArrays[i][pin-hNs:pin+hNs] += sw*yw             # overlap-add and apply a synthesis window
			pin += H                                              # advance sound pointer

		# Sum the content of the three time-domain-bandlimited output arrays into final output array
		out = outputArrays.sum(axis=0)
		# Scale down the final output (optimally I would have windowed-out for the filtering process)
		out *= 0.3

	return out
