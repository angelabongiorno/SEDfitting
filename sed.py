
import fortranformat as ff
import time
import pickle
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import astropy.units as u
from scipy import interpolate
from scipy import optimize
from itertools import product 
import os

import astrolib as al
import temp_code as tc


# Reading of data file
def read_data_file(filename, fformat):
    returnArray = []
    with open (filename, 'r') as f:
        for line in f:
            if len(line) > 1: # Parse the line only if there is more than linebreak
                returnArray.append(ff.FortranRecordReader(fformat).read(line))

    return returnArray


# Convert data array in to Astro Objects
def convert_data_to_astro_object(dataarray, cosmConstant, Omega_m):
    returnArray = []
    for element in dataArray:
        rawdata = []
        for i in range(2, len(element), 2):
            rawdata.append({'flux': float(element[i]), 'err': float(element[i+1])})

        returnArray.append(al.AstroObject(element[0], float(element[1]), pd.DataFrame(rawdata), H0=cosmConstant, Om0=Omega_m))
    
    return returnArray

# Reading of lambda file
def read_lambda_file(filename, fformat):
    tmpArray = []

    with open (filename, 'r') as f:
        for line in f:
            if len(line) > 1 and line.strip()[0] != '#':
                tmpArray.append(ff.FortranRecordReader(lambdaFormat).read(line))

    return tmpArray

def chi2_for_minimize(x, *args):
    err = 0
    obs = args[0]
    obs_err = args[1]

    library_array = []
    for i in range(2, len(args)):
        library_array.append(args[i] * x[i-2])
    
    for i in range(len(obs)):
        lib_sum = 0
        for lib in library_array:
            lib_sum += lib[i]

        err += ((obs[i] - lib_sum)**2)/(obs_err[i]*obs_err[i])

    return(err)



tic =  time.perf_counter()

with open ('tmp/input.tmp', 'r') as f:
    dataFile = f.readline().split()[0]              # Actual input file, the magnitudes for each frequency (y-axis) and errors as pair
    _ = f.readline().split()[0]                     # NOT USED progFile This is an input file that is modified according to some rules, not used at the moment
    inputFormat = f.readline().split()[0]           # Input file format for the FortranFormat
    _ = int(f.readline().split()[0])                # NOT USED nrObjects Number of objects, this is also the number of rows in the data file
    _ = int(f.readline().split()[0])                # NOT USED nrMagn Number of magnitudes. This should be the same as the number of float pairs in inputFormat
    lambdaFile = f.readline().split()[0]            # Lambdas (the frequencies) for the inputs (x-axis)
    lambdaFormat = f.readline().split()[0]          # Lambda file formt for the FortranFormat
    cosmConstant = float(f.readline().split()[0])   # Plain cosmology constant
    Omega_m = float(f.readline().split('d0')[0])    # Fortram formatted Omega_m
    _ = float(f.readline().split('d0')[0])          # NOT USED Fortran formatted Omega_l
 
    _ = f.readline().split()[0]                     # NOT USED outputFormat FortranFormat type of definiton to output format
    
    libraryNames = []
    while line := f.readline():
        if (len(line) > 1 and os.path.isfile(line.split()[0])):
            libraryNames.append(line.split()[0])

    print("Data file: "+dataFile)
    #print("Prog file: "+progFile)
    print("Input file format: "+inputFormat)
    #print("Number of objects: nrObjects)
    #print(nrMagn)
    print("Lambda file: "+lambdaFile)
    print("Lambda format: "+lambdaFormat)
    print(f"Cosmologial constant: {cosmConstant}")
    print(f"Omgea_m: {Omega_m}")
    # print(Omega_l)
    # print("Output format: "+outputFormat)

    print("Libraries to be used: ")
    for lib in libraryNames:
        print(lib)

    print("\n")

# Data file
dataArray = read_data_file(dataFile, inputFormat)

# Convert data array in to array of AstroObjects
AOArray = convert_data_to_astro_object(dataArray, cosmConstant, Omega_m)

for element in AOArray:
    element.preprocess_errors()
    # element.print()
    

lambdaArray = read_lambda_file(lambdaFile, lambdaFormat)
# print("Lambda Array:")
# print(lambdaArray)    

for element in AOArray:
    element.add_lambda(lambdaArray[0])
    # element.print()


MeasuredObject = AOArray[0] # Operate with the first measured object from input fils
MeasuredObject.normalize()

toc = time.perf_counter()
print(f"Options read in {toc-tic:0.4f} seconds.\n")



# Read all libraries in to memory

tic = time.perf_counter()

libraryArray = np.empty((1,0))
libraryPickle = "libraries.pickle"

if(os.path.isfile(libraryPickle)):
    print("Reading libraries from pickle.")
    libraryArray = pickle.load(open(libraryPickle, 'rb'))
else:
    for libname in libraryNames:
        with open(libname, 'r') as f:
            library = np.empty((1,0))
            for line in f:
                if len(line.split()) > 0:
                    library = np.append(library, al.LibraryObject(line.split()[0]) )
            libraryArray = np.append(libraryArray, al.Library(libname, library))

    print("Writing libraries to pickle.")
    pickle.dump(libraryArray, open(libraryPickle, 'wb'))

toc = time.perf_counter()
print(f"Libraries read in {toc-tic:0.4f} seconds.")



# Normalize and interpolate library objects for calculations

tic = time.perf_counter()

for lib in libraryArray:
    for object in lib.data:
        object.calc_measured_values2(MeasuredObject.data['lambda_em'])
        object.normalize()

toc = time.perf_counter()
print(f"Interpolated and normalized all library objects in {toc-tic:0.4f} seconds.")


# Calculating coefficients for library objects

methods = [
    #'Nelder-Mead',
   'Powell',           ###### allows bounds
    #'CG',
    #'BFGS',
    # 'Newton-CG',
    #'L-BFGS-B',
    'TNC',           ####### allows bounds
    #'COBYLA',
    #'SLSQP',
    'trust-constr'   ####### allows bounds
    # 'dogleg',
    # 'trust-ncg',
    # 'trust-exact',
    # 'trust-krylov'
]

initials = [0.1 for i in range(len(libraryArray))]
bounds = [(0, None) for i in range(len(libraryArray))]

agn = libraryArray[0].data[81]
gal = libraryArray[1].data[18]

for method in methods:
  res = optimize.minimize( chi2_for_minimize, initials,
                          args=(MeasuredObject.norm_data, MeasuredObject.norm_err, agn.norm_data, gal.norm_data), method=method,
                          bounds = bounds,
                          options={ "maxiter": 100000, "disp": False})

  kAGN = ( res.x[0] / agn.norm) * MeasuredObject.norm
  kGAL = ( res.x[1] / gal.norm) * MeasuredObject.norm

  print(f"\n{method=}\n{res.message}\n")
  print(f"{kAGN=}")
  print(f"{kGAL=}")




# Plotting results

plot_y = MeasuredObject.data['norm_flux']

agn.prepare_plot(kAGN)
agn_y = agn.data['norm_flux']

gal.prepare_plot(kGAL)
gal_y = gal.data['norm_flux']


# sum_object = sum_astro_data(sedAGNModelArray[81].data, sedGALModelArray[18].data, value='norm_flux')

x_max = MeasuredObject.data['lambda_em'].iloc[-1]
x_min = MeasuredObject.data['lambda_em'].iloc[0]

y_max = max(plot_y)
y_min = min(plot_y)

plt.xlim( [ x_min * 0.7, x_max * 1.3])
plt.ylim( [ y_min * 0.5, y_max * 1.5])

plt.scatter(MeasuredObject.data['lambda_em'], plot_y, c='black')
plt.plot(agn.data['lambda_em'], agn_y, color='blue')
plt.plot(gal.data['lambda_em'], gal_y, color='magenta')
# plt.plot(sum_object.data['lambda_em'], sum_object.data['norm_flux'])

plt.yscale("log")
plt.xscale("log")
plt.grid(True)
plt.show()




# fobs_err = AOArray[0].data[['errFlam']].to_numpy().flatten()
# fobs = np.multiply(AOArray[0].data['lambda'].to_numpy(), AOArray[0].data['Flam'].to_numpy())########### lambda_em---->lambda (lambda_obs)
# fAGN = np.multiply(agn.interpolated_data['Flam'].to_numpy(), agn.interpolated_data['lambda_em'].to_numpy())
# fGAL = np.multiply(gal.interpolated_data['Flam'].to_numpy(), gal.interpolated_data['lambda_em'].to_numpy()) 
# 
# fobs_max = fobs.max()
# fAGN_max = fAGN.max()
# fGAL_max = fGAL.max()
# 
# fobs_norm = fobs / fobs_max
# fobs_err_norm = fobs_err/fobs_max     ########### Normalize the errors
# fAGN_norm = fAGN / fAGN_max
# fGAL_norm = fGAL / fGAL_max
# 
# print(np.array_equiv(fobs_norm, MeasuredObject.norm_data))
# print(np.array_equiv(fobs_err_norm, MeasuredObject.norm_err))
# print(np.array_equiv(fAGN_norm, agn.norm_data))
# print(np.array_equiv(fGAL_norm, gal.norm_data))
# 
# for method in methods:
#   res = optimize.minimize( chi2_for_minimize, [0.1, 0.1], ############ use normalized fluxes and add bounds
#                           args=(fobs_norm, fobs_err_norm, fAGN_norm, fGAL_norm), method=method,
#                           bounds =((0,None),(0,None)),
#                           options={ "maxiter": 100000, "disp": False})
# 
#   
# 
#   kAGN = ( res.x[0] / fAGN_max) * fobs_max
#   kGAL = ( res.x[1] / fGAL_max) * fobs_max
# 
#   print(f"\n{method=}\n{res.message}\n")
#   print(f"{kAGN=}")
#   print(f"{kGAL=}")
# 
#   fAGN_final = fAGN * kAGN
#   fGAL_final = fGAL * kGAL
# 
#   Sya = sum((fobs*fAGN)/(fobs_err*fobs_err))
#   Saa = sum((fAGN*fAGN)/(fobs_err*fobs_err))
#   Sgg = sum((fGAL*fGAL)/(fobs_err*fobs_err))
#   Syg = sum((fobs*fGAL)/(fobs_err*fobs_err))
#   Sag = sum((fAGN*fGAL)/(fobs_err*fobs_err))
#   kAGN_true = (Sag*Syg-Sgg*Sya)/(Sag*Sag-Sgg*Saa)
#   kGAL_true = (Sag*Sya-Saa*Syg)/(Sag*Sag-Sgg*Saa)
#   
#   print(f"{kAGN_true=}")
#   print(f"{kGAL_true=}")