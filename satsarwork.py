from sarpy.io.phase_history.cphd import CPHDReader
from numpy import fft
import numpy as np
import matplotlib.pyplot as plt
from IPython import embed
from scipy.interpolate import interp1d
import scipy
import math

pulse_start = 50
pulse_end = 1000

reader = CPHDReader("2023-11-09-12-38-46_UMBRA-04_CPHD.cphd")
cphd_data = reader.read(slice(pulse_start,pulse_end),None)
num_pulses = reader.cphd_meta.Data.Channels[0].NumVectors
samples_per_pulse = reader.cphd_meta.Data.Channels[0].NumSamples

print(f"Data set has {num_pulses} pulses with {samples_per_pulse} frequency samples per pulse")
#(1950,15999) Pulse x Frequency Bins

datacube = reader.read(slice(pulse_start,pulse_end),None)
print(datacube.shape)
taper= scipy.signal.windows.taylor(pulse_end-pulse_start)[:,None]*scipy.signal.windows.taylor(samples_per_pulse)[None,:]
rd = fft.ifftshift(fft.ifft2(fft.ifftshift(cphd_data*taper))) #shift DC to center, invert2d, shift 2dfft #range doppler
rd_db = 20*np.log10(np.abs(rd))

'''            
plt.imshow(rd_db, aspect = 'auto',cmap = 'gray')
plt.clim([-50,-20])
plt.show()
'''

# pulse range
rp = fft.ifftshift(fft.ifft(fft.ifftshift(datacube*taper, axes = 1), axis = 1),axes = 1) #shift DC to center, invert2d, shift 2dfft 

#plt.imshow(rp_db, aspect = 'auto')
#plt.show()
#plt.plot(rp_db[0])
#plt.show()


pvp = reader.read_pvp_block()
iarp = reader.cphd_meta.SceneCoordinates.IARP
aimpoint= np.array([iarp.ECF.X,iarp.ECF.Y,iarp.ECF.Z,]) #focus point xyz



vp = pvp["Primary"]
rxpos = vp['RcvPos'] #receiver pos
txpos = vp['TxPos'] #transmitter pos
ss = vp["SCSS"] #sample spacing
s0 = vp["SC0"] #first freq bin


sEnd = s0 + (samples_per_pulse-1) * ss
bandwidth = sEnd - s0 # num freq bins * bin spacing
center_freq = (s0 + sEnd) / 2 #starting freq + half bandwidth is center freq
center_freq = center_freq[0]

spd = 299792458
rr = spd/(2*bandwidth) #range resolution
num_bins = rp.shape[1]


positions = (rxpos + txpos) / 2
aimNorm=aimpoint/np.linalg.norm(aimpoint)
midx = positions[((pulse_end - pulse_start) // 2),:] #midpoint between tx and rx for every position
midnorm = (midx - aimpoint)/np.linalg.norm(midx - aimpoint) # vector to sat from aimpiint
rangenorm = midnorm-(midnorm.dot(aimNorm)*aimNorm) # normalized range vector
cross_range  = np.cross(aimNorm, rangenorm) # normalized cross range vector

grid_extent = 4000  # meters
u = np.linspace(-grid_extent, grid_extent, grid_extent//2)
v = np.linspace(-grid_extent, grid_extent, grid_extent//2)
meshrange, meshcrossrange = np.meshgrid(u, v)



points_x = np.array(meshrange * rangenorm[0] + meshcrossrange * cross_range[0] + aimpoint[0])
points_y = np.array(meshrange* rangenorm[1] + meshcrossrange * cross_range[1] + aimpoint[1])
points_z = np.array(meshrange * rangenorm[2] + meshcrossrange * cross_range[2] + aimpoint[2])

image = np.zeros(meshrange.shape, dtype=np.complex128)
print(rp.shape[0])
for i in range(pulse_start, pulse_end):
    curpos=positions[i,:]
    curpulse = rp[i-pulse_start,:]
    range=np.linalg.norm(curpos-aimpoint)
    R = np.sqrt(
    ((points_x - curpos[0]) ** 2 + 
     (points_y - curpos[1]) ** 2 +
     (points_z - curpos[2]) ** 2) 
    )
    rangediff=R-range
    index=rangediff/rr[0]+num_bins//2
    phaser= np.exp(1j *(4*math.pi*rangediff*center_freq/spd))
    
    # Interpolate across entire 2D range matrix
    intp = curpulse[index.astype(int)]
    intp_complex = intp.astype(np.complex128)
    intp_complex*=phaser
    image += intp_complex

image = 20*np.log10(np.abs(image)+.00000000001)

plt.imshow(image, aspect = 'equal', origin='lower', cmap = 'plasma')
plt.xlabel('Image X Position (m)')
plt.ylabel('Image Y Position (m)')
plt.title('Backprojected Image')
plt.colorbar(label ='Intensity (dB)')
plt.show()


reader.close()