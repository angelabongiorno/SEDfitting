from astropy.cosmology import FlatLambdaCDM
import astropy.units as u
from astropy.units import cds

from scipy import interpolate
from scipy import optimize
from itertools import product 

import pandas as pd
import numpy as np
import math
from scipy import interpolate

AO_ERR_LOW = 0.0
AO_ERR_HIGH = 0.05

# C = 2.99792458e18 # ångström/s
C = cds.c.to(u.angstrom / u.s)

class AstroObject:
    """
    AstroObject contains data of measured object

    Attributes:
    id: str
        Identifier of the measured object.
    redshift : float
        The redshift value of the measured object.
    data: pandas.core.frame.DataFrame
        All measured and calculated data of the measured object.
    dl: float
        Calculated luminosity distance in Mpc for the redshift and constructor parameters.
    """

    def __init__(self, id, redshift, data, H0=67.32, Om0=0.3158, Tcmb0=2.725):
        """
        Parameters:
        id: str
            Identifier of the measured object.
        redshift : float
            The redshift value of the measured object.
        data: pandas.core.frame.DataFrame
            All measured and calculated data of the measured object.
        H0: float or Astropy quantity
            Hubble constant at z = 0. If a float, must be in [km/sec/Mpc]. Default 67.32.
        Om0: float
            Omega matter: density of non-relativistic matter in units of the
            critical density at z=0. Default 0.3158.
        Tcmb0: float
            Temperature of the CMB z=0. If a float, must be in [K]. Default: 0 [K].
            Setting this to zero will turn off both photons and neutrinos (even massive ones).
            Default 2.725.
        """
        self.id = id
        self.redshift = redshift
        self.data = data
        self.dl = FlatLambdaCDM(
            H0=H0 * u.km / u.s / u.Mpc, Tcmb0=Tcmb0 * u.K, Om0=Om0).luminosity_distance(redshift).value * 3.08568e24 #in centimeters

        # Set values outside 10 - 40 as zero
        self.data['flux'] = [ x if 10 < x < 40 else 0 for x in self.data['flux'] ]

    def print(self):
        """
        Prints all parameters (id, redshift, luminosity distance and data) to standard output.
        """
        print(f"Object id {self.id}")
        print(f"Redshift {self.redshift}")
        print(f"Luminosity distance {self.dl}")
        print(f"Object data:\n{self.data}")

    def preprocess_errors(self, err_low_limit=AO_ERR_LOW, err_high_limit=AO_ERR_HIGH):
        """
        Making sure that measurement errors fall between desired range.
        Parameters:
        err_low_limit : float
            The low limit of the error, defaults to 0.0.
        err_high_limit : float
            The high limit of the error, defaults to 0.05.
        """
        for i in range(len(self.data['err'])):
        #   if err_low_limit <= self.data['err'][i]  < err_high_limit:
        #       self.data['err'][i] = math.sqrt(0.05**2+(0.1*0.05)**2) # TODO: Why was this? 
            if self.data['err'][i]  < err_low_limit:
                self.data['err'][i] = np.sqrt(err_low_limit**2+( self.data['err'][i])**2) 

    def add_lambda(self, lambdaArray):
        """
        Adds array of lambdas to the object. Calculates automatically 'freq', 'lambda_em', 'freq_em' and 'Fni' values 
        to the self.data attribute.
        Parameters:
        lambdaArray : pandas.core.frame.DataFrame
            The Dataframe that contains the Lambdas
        """
        self.data['lambda'] = lambdaArray
        self.data['freq']= [cds.c.to(u.angstrom / u.s) / elem for elem in lambdaArray]
        self.data['lambda_em'] = [ lam / (1+self.redshift) for lam in lambdaArray]
        self.data['freq_em'] = [freq * (1+self.redshift) for freq in self.data['freq']]
        self.data['Fni'] = [ 10**(-0.4*(flux+48.60)) for flux in self.data['flux']] # TODO: plain numbers in code, why?
        
        errFni = []
        for i in range(len(self.data['Fni'])):
            errFni.append(0.92 * self.data['Fni'][i] * self.data['err'][i])
        self.data['errFni'] = errFni
        
        Flam = []
        for i in range(len(self.data['Fni'])):
            Flam.append( self.data['Fni'][i] * cds.c.to(u.angstrom / u.s) / (self.data['lambda'][i]**2))
            # Flam.append( self.data['Fni'][i] * cds.c.to(u.angstrom / u.s) / (self.data['lambda'][i]))
        self.data['Flam'] = Flam
        
        errFlam = []
        for i in range(len(self.data['errFni'])):
            errFlam.append( self.data['errFni'][i] * cds.c.to(u.angstrom / u.s)/(self.data['lambda'][i]**2))
            # errFlam.append( self.data['errFni'][i] * cds.c.to(u.angstrom / u.s)/(self.data['lambda'][i]))
        self.data['errFlam'] = errFlam

        # self.data['freq_em_log'] = [ math.log10(i) for i in self.data['freq_em']]
        # self.data['lambda_em_log'] = [ math.log10(i) for i in self.data['lambda_em']]
        # self.data['Fni_log'] = [ math.log10(i) for i in self.data['Fni']]
        # self.data['errFni_log'] = [ math.log10(i) for i in self.data['errFni']]
        self.data['Flam_log'] = [ math.log10(i) for i in self.data['Flam']]
        # self.data['errFlam_log'] = [ math.log10(i) for i in self.data['errFlam']]

        for i in range (len(self.data)):
            if self.data['flux'][i] == 0:
                self.data['Fni'][i] = 0;
                self.data['errFni'][i] = 0;
                self.data['Flam'][i] = 0;
                self.data['errFlam'][i] = 0;
                # self.data['Fni_log'][i] = 0;
                # self.data['errFni_log'][i] = 0;
                self.data['Flam_log'][i] = 0;
                # self.data['errFlam_log'][i] = 0;
                
        plot_y = []
        for i in range(len(self.data['Flam_log'])):
            plot_y.append(self.data['lambda'][i] * self.data['Flam'][i] )
        self.data['norm_flux'] = plot_y   

    def normalize(self):
        # fobs = np.multiply(self.data['lambda'].to_numpy(), self.data['Flam'].to_numpy())
        fobs = self.data['Flam'].to_numpy()

        fobs_err = self.data[['errFlam']].to_numpy().flatten()

        self.norm = fobs.max()
        self.norm_data = fobs / self.norm
        self.norm_err = fobs_err / self.norm


class LibraryObject:
    """
    AstroObject contains data of measured object

    Attributes:
    id: str
        Identifier of the measured object.
    redshift : float
        The redshift value of the measured object.
    data: pandas.core.frame.DataFrame
        All measured and calculated data of the measured object.
    dl: float
        Calculated luminosity distance in Mpc for the redshift and constructor parameters.
    """

    def __init__(self, filename, data=None, lazyload = False):
        """
        Parameters:
        filename: str
            Identifier of the library object as filename pointing to the data. 
        data: pandas.core.frame.DataFrame
            All data of the library object. At least Flam_log column is required if supplied
        lazyload: Boolean
            Whether to load the file data immediately or when used.
        """
        self.id = filename

        if data is None:
            self.load_data()
        else:
            self.set_data(data)


    def prepare_plot(self, coefficient = 1):
        plot_y = []
        for i in range(len(self.data['Flam_log'])):
            plot_y.append(self.data['Flam'][i] * self.data['lambda_em'][i] * coefficient)
            # plot_y.append(self.data['Flam'][i] * coefficient)
        self.data['norm_flux'] = plot_y
        

    def load_data(self):
        with open (self.id, 'r') as mf:
            tempModel = []
            for mline in mf:
               values = mline.split()
               #print(f"Freq: {values[0]}, Value: {values[1]}")
               tempModel.append({ 'lambda_em': float(values[0]), 'Flam_log': float(values[1]) })
            self.set_data(pd.DataFrame(tempModel)) 


    def set_data(self, data):
        self.data = data
        if ('Flam_log' in self.data):
            self.data['Flam_log'] = [ x if x > 0 else 0 for x in self.data['Flam_log'] ]
            self.data['Flam'] = [ 10 ** x for x in self.data['Flam_log']]
            # self.data['Flam'] = self.data['Flam'] * self.data['lambda_em']

        else:
            print(f"ERROR: model {self.id} missing required data.")
            print(self.data.head())


    def print(self):
        """
        Prints all parameters (id and data) to standard output.
        """
        print(f"Library object id: {self.id}")
        print(f"Data: \n{self.data}")

    def calc_measured_values(self, lambda_em):
        """
        Interpolates values corresponding to the measures object, to which the library object is
        compared to. Creates new attribute interpolated_data, which contains 'labda_em' and 'Flam' 
        columns. 'Lambda_em' is copied from function parameter and 'Flam' is interpolated from the 
        model data using Scipy interpolate.interp1d() function.
        
        Parameters:
        lambda em: pandas.core.frame.DataFrame
            Array holding all the measured values which needs values for Chi2 error calculation.
        """
        temp_model = []

        for ao_le in lambda_em:
            lower,  higher = (0, 0) # variables to store lower and higher lambda_em for item
            for i in self.data['lambda_em']:
                if i < ao_le and i > lower:
                    lower= i
                if i > ao_le:
                    higher = i
                    break # for loop

            low_element = self.data.loc[self.data['lambda_em'] == lower]
            high_element = self.data.loc[self.data['lambda_em'] == higher]

            # print(f"Low element:\n {low_element} \n {float(low_element['Flam'])}")
            # print(f"High element:\n {high_element}")

            # print(f"Original lambda: {ao_le}")
            # print(f" lower value: {lower}: {float(low_element['Flam'])}")
            # print(f" higher value: {higher}: {float(high_element['Flam'])}.")

            #interp = interpolate.interp1d( [lower, higher], [ float(low_element['Flam']),float(high_element['Flam']) ])
            interp = float(low_element['Flam']) +((float(high_element['Flam'])-float(low_element['Flam']))/(higher-lower))*(ao_le-lower)
            # print(f"interpolated Flam: {float(interp(ao_le))}")       

            #temp_model.append({'lambda_em': ao_le,
                               #'Flam':  float(interp(ao_le)) })
            temp_model.append({'lambda_em': ao_le,
                               'Flam':  float(interp)})
        # print(temp_model)
        self.interpolated_data = pd.DataFrame(temp_model)
        
    def normalize(self):
        if (hasattr(self, 'interpolated_data') and 'Flam' in self.interpolated_data):
            # f = np.multiply(self.interpolated_data['Flam'].to_numpy(), self.interpolated_data['lambda_em'].to_numpy())
            f = self.interpolated_data['Flam'].to_numpy()
            self.norm = f.max()
            self.norm_data = f / self.norm
        else:
            print(f"ERROR: model {self.id} missing required data (Flam).")
            print(self.data.head())

    
    def calc_measured_values2(self, lambda_em):
        """
        Interpolates values corresponding to the measures object, to which the library object is
        compared to. Creates new attribute interpolated_data, which contains 'labda_em' and 'Flam' 
        columns. 'Lambda_em' is copied from function parameter and 'Flam' is interpolated from the 
        model data using Scipy interpolate.interp1d() function.
        
        Parameters:
        lambda em: pandas.core.frame.DataFrame
            Array holding all the measured values which needs values for Chi2 error calculation.
        """
        temp_model = []

        if ('lambda_em' in self.data):
            for ao_le in lambda_em:
                low_index,  high_index = (0, len(self.data['lambda_em'])) # variables to store the index of the tempalte array
                if (ao_le <self.data['lambda_em'].iloc[low_index]) or (ao_le>self.data['lambda_em'].iloc[high_index-1]):
                   temp_model.append({'lambda_em': ao_le,
                                   'Flam':  0}) 
                    
                else:
                     while True:
                          if high_index-low_index ==1 or high_index-low_index==0:
                              lower = self.data['lambda_em'].iloc[high_index-1]
                              higher = self.data['lambda_em'].iloc[high_index]
                              low_element = self.data['Flam'].iloc[high_index-1]
                              high_element = self.data['Flam'].iloc[high_index]
                              break
                              
                          middle_index = int((high_index+low_index)/2)
                          if ao_le < self.data['lambda_em'].iloc[middle_index]:
                               high_index = middle_index
                          elif ao_le >= self.data['lambda_em'].iloc[middle_index]:
                               low_index = middle_index
                
                     interp = float(low_element) +((float(high_element)-float(low_element))/(higher-lower))*(ao_le-lower)
                     temp_model.append({'lambda_em': ao_le,
                                   'Flam':  float(interp)})
            # print(temp_model)
            self.interpolated_data = pd.DataFrame(temp_model)
        else:
            print(f"ERROR: model {self.id} missing required data (lambda_em).")
            print(self.data.head())

class Library:
    def __init__(self, filename, data):
        self.name = filename
        self.data = data

    def findLibraryObject(self, id):
        for item in self.data:
            if (item.id == id):
                return item



class Result:
    def __init__(self, library_objects, chi2, red_chi2, coefficients):
        self.library_objects = library_objects
        self.chi2 = chi2
        self.red_chi2 = red_chi2
        self.coefficients = coefficients

    def to_str(self):
        str = ""

        for obj in self.library_objects:
            str += obj.id+", "

        str += f"{self.chi2}, {self.red_chi2}, "

        for coef in self.coefficients:
            str += f"{coef}, "

        str = str[:len(str)-2]

        return str


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


def optimize_coefficients(object, library_objects):
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

    input_norm_data = [ i.norm_data for i in library_objects ]

    initials = [0.1 for i in range(len(library_objects))]
    bounds = [(0, None) for i in range(len(library_objects))]

    for method in methods:  
        res = optimize.minimize( chi2_for_minimize, initials,
                          args=(object.norm_data, object.norm_err, *input_norm_data), method=method,
                          bounds = bounds,
                          options={ "maxiter": 100000, "disp": False})

    coefficient_vector = res.x

    chi2 = chi2_for_minimize(coefficient_vector, object.norm_data, object.norm_err, *input_norm_data)
    red_chi2 = chi2 / ( len(object.norm_data) - len(library_objects))
    
    for i in range(len(coefficient_vector)):
        coefficient_vector[i] = coefficient_vector[i] / library_objects[i].norm * object.norm

    return(Result(library_objects, chi2, red_chi2, coefficient_vector))


