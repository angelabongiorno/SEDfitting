
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

#def chi2_for_computation(fobs, err, fluxes, norms):
    #for i in 
    #chi = sum(((fobs-fagn-fgal)/err)**2)
    #return(chi)


tic =  time.perf_counter()

with open ('tmp/input.tmp', 'r') as f:
    dataFile = f.readline().split()[0]              # Actual input file, the magnitudes for each frequency (y-axis) and errors as pair
    progFile = f.readline().split()[0]              # NOT USED This is an input file that is modified according to some rules, not used at the moment
    inputFormat = f.readline().split()[0]           # Input file format for the FortranFormat
    nrObjects = int(f.readline().split()[0])        # NOT USED Number of objects, this is also the number of rows in the data file
    nrMagn = int(f.readline().split()[0])           # NOT USED Number of magnitudes. This should be the same as the number of float pairs in inputFormat
    lambdaFile = f.readline().split()[0]            # Lambdas (the frequencies) for the inputs (x-axis)
    lambdaFormat = f.readline().split()[0]          # Lambda file formt for the FortranFormat
    cosmConstant = float(f.readline().split()[0])  ################ int--->float   # Plain cosmology constant
    Omega_m = float(f.readline().split('d0')[0])    # Fortram formatted Omega_m
    Omega_l = float(f.readline().split('d0')[0])    # Fortran formatted Omega_l
 
    outputFormat = f.readline().split()[0]
    sedAGNModelName = f.readline().split()[0]
    sedAGNModelNr = f.readline().split()[0]         # NOT USED Number of AGN models in library
    sedGALModelName = f.readline().split()[0]
    sedGALModelNr = f.readline().split()[0]         # NOT USED Number of GAL models in library

    print("Data file: "+dataFile)
    #print("Prog file: "+progFile)
    print("Input file format: "+inputFormat)
    #print("Number of objects: nrObjects)
    #print(nrMagn)
    print("Lambda file: "+lambdaFile)
    print("Lambda format: "+lambdaFormat)
    print(f"Cosmologial constant: {cosmConstant}")
    print(f"Omgea_m: {Omega_m}")
    #print(Omega_l)
    print("Output format: "+outputFormat)
    print("AGN library name: "+sedAGNModelName)
    # print(sedAGNModelNr)
    print("GAL library name: "+sedGALModelName)
    #print(sedGALModelNr)





# Data file
dataArray = read_data_file(dataFile, inputFormat)

# Logging
# print("Data array: ")
# print(dataArray)
# print(f"Data array objects: {len(dataArray)}, should be equal to {nrObjects}")
if len(dataArray) != nrObjects:
    print("\nERROR! Data objects in file do not match to input file number of objects.\n")

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

toc = time.perf_counter()
print(f"Options read in {toc-tic:0.4f} seconds")


tic = time.perf_counter()

sedAGNModelArray = []
AGNPickle = "AGN.pickle"
if (Path(AGNPickle).exists()):
    print("Reading AGN from pickle.")
    sedAGNModelArray = pickle.load(open(AGNPickle, "rb"))
else:
    with open (sedAGNModelName, 'r') as f:
        for fline in f:
            if len(fline.split()) > 0:
                sedAGNModelArray.append( al.LibraryObject(fline.split()[0]) )
    print("Writing AGN to pickle.")
    pickle.dump(sedAGNModelArray, open(AGNPickle, "wb"))
#sedAGNModelArray[0].print()
                    
toc = time.perf_counter()
print(f"AGN models read in {toc-tic:0.4f} seconds")


tic = time.perf_counter()
sedGALModelArray = []

GALPickle = "GAL.pickle"
if (Path(GALPickle).exists()):
    print("Reading GAL from pickle.")
    sedGALModelArray = pickle.load(open(GALPickle, "rb"))
    #print(sedGALModelArray)
else:
    with open (sedGALModelName, 'r') as f:
        for fline in f:
            if len(fline.split()) > 0:
                sedGALModelArray.append( al.LibraryObject(fline.split()[0]) )
    print("Writing GAL to pickle.")
    pickle.dump(sedGALModelArray, open(GALPickle, "wb"))
# sedGALModelArray[0].print()

toc = time.perf_counter()
print(f"GAL models read in {toc-tic:0.4f} seconds.")

tic = time.perf_counter()
for AGN_model in sedAGNModelArray:
        AGN_model.normalize()
toc = time.perf_counter()
print(f"AGN templates normalized in {toc-tic:0.4} seconds.")


tic = time.perf_counter()
for GAL_model in sedGALModelArray:
    if GAL_model.id !=  'LIBRARY/SED_GAL/BC_EXT/bcCHAB_tau30_4.0Gyr.st_4':
       GAL_model.normalize()
toc = time.perf_counter()
print(f"Galaxy templates normalized in {toc-tic:0.4} seconds.")

tic =time.perf_counter()
for GAL_model in sedGALModelArray:
    if GAL_model.id !=  'LIBRARY/SED_GAL/BC_EXT/bcCHAB_tau30_4.0Gyr.st_4':
        GAL_model.calc_measured_values2(AOArray[0].data['lambda_em']) 
toc = time.perf_counter()
print(f"Galaxy templates interpolated in {toc-tic:0.4} seconds.")
                        

AOArray[0].print()
plot_y = []
for i in range(len(AOArray[0].data['Flam_log'])):
    plot_y.append( AOArray[0].data['lambda'][i] *  AOArray[0].data['Flam'][i] )
AOArray[0].data['norm_flux'] = plot_y

print(f"AGN model: {sedAGNModelArray[81].id}.")
plot_y_agn = []
for i in range(len(sedAGNModelArray[81].data['Flam_log'])):
    # plot_y_agn.append((10 ** sedAGNModelArray[81].data['Flam_log'][i]) * (10 ** (-57.05)) * sedAGNModelArray[81].data['lambda_em'][i])
    plot_y_agn.append(sedAGNModelArray[81].data['Flam'][i] * (10 ** (-57.05)) * sedAGNModelArray[81].data['lambda_em'][i])
sedAGNModelArray[81].data['norm_flux'] = plot_y_agn


print(f"GAL model: {sedGALModelArray[18].id}.")
plot_y_gal = []
for i in range(len(sedGALModelArray[18].data['Flam_log'])):
    # plot_y_gal.append((10**sedGALModelArray[18].data['Flam_log'][i])  * (10**-23.83) * sedGALModelArray[18].data['lambda_em'][i])
    plot_y_gal.append(sedGALModelArray[18].data['Flam'][i]  * (10**-23.83) * sedGALModelArray[18].data['lambda_em'][i])
sedGALModelArray[18].data['norm_flux'] = plot_y_gal

# sum_object = sum_astro_data(sedAGNModelArray[81].data, sedGALModelArray[18].data, value='norm_flux')

x_max = AOArray[0].data['lambda_em'].iloc[-1]
x_min = AOArray[0].data['lambda_em'].iloc[0]

y_max = max(plot_y)
y_min = min(plot_y)

plt.xlim( [ x_min * 0.7, x_max * 1.3])
plt.ylim( [ y_min * 0.5, y_max * 1.5])

#plt.scatter(AOArray[0].data['lambda_em'], plot_y, c='black')
#plt.plot(sedAGNModelArray[81].data['lambda_em'], plot_y_agn, color='blue')
#plt.plot(sedGALModelArray[18].data['lambda_em'], plot_y_gal, color='magenta')
# plt.plot(sum_object.data['lambda_em'], sum_object.data['norm_flux'])

plt.yscale("log")
plt.xscale("log")
plt.grid(True)
# plt.show()

sedAGNModelArray[81].calc_measured_values( AOArray[0].data['lambda_em'] )
# print(sedAGNModelArray[81].interpolated_data)

interpolated_AGN_y = []
for i in range(len(sedAGNModelArray[81].interpolated_data['Flam'])):
    interpolated_AGN_y.append(sedAGNModelArray[81].interpolated_data['Flam'][i] * (10 ** (-57.05)) * sedAGNModelArray[81].interpolated_data['lambda_em'][i])

# plt.scatter(sedAGNModelArray[81].interpolated_data['lambda_em'], interpolated_AGN_y, color='yellow')



sedGALModelArray[18].calc_measured_values( AOArray[0].data['lambda_em'] )
# print(sedGALModelArray[18].interpolated_data)

interpolated_GAL_y = []
for i in range(len(sedGALModelArray[18].interpolated_data['Flam'])):
    interpolated_GAL_y.append(sedGALModelArray[18].interpolated_data['Flam'][i] * (10 ** (-23.83)) * sedGALModelArray[18].interpolated_data['lambda_em'][i])

# plt.scatter(sedGALModelArray[18].interpolated_data['lambda_em'], interpolated_GAL_y, color='green')

# summed_object = sum_library_objects([sedAGNModelArray[81].interpolated_data.to_numpy(), sedGALModelArray[18].interpolated_data.to_numpy()])
measured_object = AOArray[0].data[['lambda_em', 'Flam', 'errFlam']].to_numpy()

# print(summed_object)
# print(measured_object)

#Testing witht the Chi2

fobs_err = AOArray[0].data[['errFlam']].to_numpy().flatten()

# print(f"FOBS_err: {fobs_err}")

fobs = np.multiply(AOArray[0].data['lambda'].to_numpy(), AOArray[0].data['Flam'].to_numpy())########### lambda_em---->lambda (lambda_obs)
fAGN = np.multiply(sedAGNModelArray[81].interpolated_data['Flam'].to_numpy(), sedAGNModelArray[81].interpolated_data['lambda_em'].to_numpy())
fGAL = np.multiply(sedGALModelArray[18].interpolated_data['Flam'].to_numpy(), sedGALModelArray[18].interpolated_data['lambda_em'].to_numpy()) 

# chi2_err = calc_chi2_err(fobs, fobs_err, [fAGN, fGAL])
# print(f"Err before: {chi2_err}")
# 
# fAGN *=  math.sqrt((10 ** (-57.05)))
# fGAL *=  math.sqrt((10 ** (-23.83)))
# 
# chi2_err = calc_chi2_err(fobs, fobs_err, [fAGN, fGAL])
# print(f"Err in the middle: {chi2_err}")
# 
# fAGN *=  math.sqrt((10 ** (-57.05)))
# fGAL *=  math.sqrt((10 ** (-23.83)))
# 
# chi2_err = calc_chi2_err(fobs, fobs_err, [fAGN, fGAL])
# print(f"Err after: {chi2_err}")


# Calculate the max values for normalizing
fobs_max = fobs.max()
fAGN_max = fAGN.max()
fGAL_max = fGAL.max()


# Normalize the fluxes
fobs_norm = fobs / fobs_max
fobs_err_norm = fobs_err/fobs_max     ########### Normalize the errors
fAGN_norm = fAGN / fAGN_max
fGAL_norm = fGAL / fGAL_max



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

        err += ((obs[i] - lib_sum)**2)/(obs_err[i]*obs_err[i])   ########## errors^2

    return(err)

methods = [
    #'Nelder-Mead',
   'Powell',           ###### allows bounds
    #'CG',
    #'BFGS',
    # 'Newton-CG',
    #'L-BFGS-B',
    #'TNC',           ####### allows bounds
    #'COBYLA',
    #'SLSQP',
    #'trust-constr'   ####### allows bounds
    # 'dogleg',
    # 'trust-ncg',
    # 'trust-exact',
    # 'trust-krylov'
]





for method in methods:
  res = optimize.minimize( chi2_for_minimize, [0.1, 0.1], ############ use normalized fluxes and add bounds
                          args=(fobs_norm, fobs_err_norm, fAGN_norm, fGAL_norm), method=method,
                          bounds =((0,None),(0,None)),
                          options={ "maxiter": 100000, "disp": False})

  

  kAGN = ( res.x[0] / fAGN_max) * fobs_max
  kGAL = ( res.x[1] / fGAL_max) * fobs_max

  print(f"\n{method=}\n{res.message}\n")
  print(f"{kAGN=}")
  print(f"{kGAL=}")

  fAGN_final = fAGN * kAGN
  fGAL_final = fGAL * kGAL

  plt.scatter(sedAGNModelArray[81].interpolated_data['lambda_em'], fAGN_final+fGAL_final, color='yellow')
  #plt.scatter(sedGALModelArray[18].interpolated_data['lambda_em'], fGAL_final, color='green')
  plt.pause(0.05)
  
  
  Sya = sum((fobs*fAGN)/(fobs_err*fobs_err))
  Saa = sum((fAGN*fAGN)/(fobs_err*fobs_err))
  Sgg = sum((fGAL*fGAL)/(fobs_err*fobs_err))
  Syg = sum((fobs*fGAL)/(fobs_err*fobs_err))
  Sag = sum((fAGN*fGAL)/(fobs_err*fobs_err))
  kAGN_true = (Sag*Syg-Sgg*Sya)/(Sag*Sag-Sgg*Saa)
  kGAL_true = (Sag*Sya-Saa*Syg)/(Sag*Sag-Sgg*Saa)
  
  print(f"{kAGN_true=}")
  print(f"{kGAL_true=}")

  
#
# 
# plt.scatter(sedAGNModelArray[81].interpolated_data['lambda_em'], fAGN_final, color='yellow')
# plt.scatter(sedGALModelArray[18].interpolated_data['lambda_em'], fGAL_final, color='green')
# plt.scatter(sedGALModelArray[18].interpolated_data['lambda_em'], fGAL + fAGN, color='black')


plt.show()

