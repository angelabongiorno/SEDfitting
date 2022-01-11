from astropy.cosmology import FlatLambdaCDM
import astropy.units as u
from astropy.units import cds

import pandas as pd
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
            if err_low_limit <= self.data['err'][i]  < err_high_limit:
                self.data['err'][i] = math.sqrt(0.05**2+(0.1*0.05)**2) # TODO: Why was this?

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
        self.data['Flam'] = Flam
        
        errFlam = []
        for i in range(len(self.data['errFni'])):
            errFlam.append( self.data['errFni'][i] * cds.c.to(u.angstrom / u.s)/(self.data['lambda'][i]**2))
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

    def __init__(self, id, data):
        """
        Parameters:
        id: str
            Identifier of the library object. 
        data: pandas.core.frame.DataFrame
            All data of the library object. At least Flam_log column is required.
        """
        self.id = id
        self.data = data
        # col_str = ""
        # for col in self.data.columns:
        #     col_str += " " + str(col)
        # print(col_str)
        
        if ('Flam_log' in self.data):
            self.data['Flam_log'] = [ x if x > 0 else 0 for x in self.data['Flam_log'] ]
            self.data['Flam'] = [ 10 ** x for x in self.data['Flam_log']]
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

            interp = interpolate.interp1d( [lower, higher], [ float(low_element['Flam']),float(high_element['Flam']) ])

            # print(f"interpolated Flam: {float(interp(ao_le))}")       

            temp_model.append({'lambda_em': ao_le,
                               'Flam':  float(interp(ao_le)) })
        # print(temp_model)
        self.interpolated_data = pd.DataFrame(temp_model)

