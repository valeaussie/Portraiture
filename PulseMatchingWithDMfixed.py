import bilby
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

plt.close('all')

label = "gaussian_with_dm"
outdir = "outdir"

# Constant for DM delay calculat8ion
K_DM = 4.15e3  # MHz^2*pc^-1*cm^3 [1/s^2 * pc^-1*cm^3 ]
freqs = [700,750,800,1000]  # Central frequencies for two bands in MHz

def plot_it(modeled_signals,datasets):
    plt.figure()
    for k in range(3):
        for band in range(len(freqs)):
            plt.subplot(4, 3, k + band*3 + 1)
            plt.plot(x, datasets[k, band], "-o", label=f"Data (Signal {k+1}, Band {band+1})")
            plt.plot(x, modeled_signals[k, band], label=f"Modeled (Signal {k+1}, Band {band+1})")
            if k==0:
                plt.legend()
    plt.tight_layout()
    plt.show()

def create_lorentzian(n_points=100, mean=0.5, gamma=0.05):
    x = np.linspace(0, 1, n_points)
    mid_l = 1 / (1 + ((x - 0.5) / gamma) ** 2)
    # Lorentzian function: f(x) = 1 / (1 + ((x - mean) / gamma) ** 2)
    lorentzian = translate_signal(mid_l, shift=mean-0.5)
    
    return x, lorentzian


def create_gaussian(n_points=100, mean=0.5, sigma=0.05):
    
    x = np.linspace(0, 1, n_points)
    middle_gauss = np.exp(-0.5 * ((x - 0.5) / sigma) ** 2)
    gaussian = translate_signal(middle_gauss, shift=mean-0.5)
    
    return x, gaussian

# Function to apply a DM-induced delay
def apply_dm_delay(signal, dm, freq, n_points):
    #[1/s^2 * pc^-1*cm^3 ] * pc/cm^3 = 1/s^2
    delay = int(np.round(K_DM * dm / freq**2 * n_points))  # Delay in samples
    delay = (delay%n_points)/n_points
    return translate_signal(signal, delay)

def invert_dm_delay(signal, dm, freq, n_points):
    delay = int(np.round(K_DM * dm / freq**2 * n_points))  # Delay in samples
    delay = (delay%n_points)/n_points
    return translate_signal(signal, -delay)


# Function to translate a signal in time
def translate_signal(signal, shift):
    
    n = len(signal)
    shift = int(shift*n)
    
    shifted_signal = np.zeros_like(signal)
    if shift > 0:
        shifted_signal[shift:] = signal[:n - shift]
        shifted_signal[:shift] = signal[n - shift:]
    elif shift < 0:
        shift = -shift
        shifted_signal[:n - shift] = signal[shift:]
        shifted_signal[n - shift:] = signal[:shift]
    else:
        shifted_signal = signal.copy()
    return shifted_signal

# Function to generate multiband datasets
def generate_datasets():
    n_points = 100
    means = [0.10, 0.35, 0.7]  # Different means for each Gaussian  
    dms = [10,10.02,10.03]  # Dispersion measure in pc/cm^3

    datasets = []
    for mean,dm in zip(means,dms):
            
        bands=[]
        for f in np.arange(len(freqs)):
            freq = freqs[f]
            if f==0:
                _, signal  = create_gaussian(n_points=n_points, mean=mean)  
            if f==1:
                _, signal  = create_lorentzian(n_points=n_points, mean=mean)  
            if f==2:
                _, signal  = create_gaussian(n_points=n_points, mean=mean,sigma=0.1)
            if f==3:
                _, signal  = create_lorentzian(n_points=n_points, mean=mean,gamma=0.1)  
                
            bands.append(apply_dm_delay(signal, dm, freq, n_points))            
        
        datasets.append(bands)
    return np.array(datasets)  # Shape: (3 signals, 2 bands, n_points)

# Function to align and average signals
def align_and_average(signals, d1, d2, target_index):
   
    aligned_signal1 = translate_signal(signals[0], -d1)
    aligned_signal2 = translate_signal(signals[1], -d2)
   
    
    template = (aligned_signal1 + aligned_signal2) / 2
    return template

# Model function to include DM
def model_signal(dm0,dm1,dm2, d0, d1, d2,datasets):
    n_points = datasets.shape[2]
    modeled_signals = []
    
    diffs = [d0, d1, d2]
    dms=[dm0,dm1,dm2]

    for i in range(3): # run over pulses 
        idx1, idx2 = [(i + 1) % 3, (i + 2) % 3]
        if idx1 > idx2:
            idx1, idx2 = idx2, idx1

        d1, d2 = diffs[idx1], diffs[idx2]
        dm1, dm2 = dms[idx1], dms[idx2]
        templates=[]
        for f in np.arange(len(freqs)):
            templates.append(align_and_average([invert_dm_delay(datasets[idx1,f], dm1, freqs[f], n_points),
                                                invert_dm_delay(datasets[idx2,f], dm2, freqs[f], n_points)],
                                                d1, d2, i))

        # Apply DM effect to both bands
        band_signals = []
        for freq,f in zip(freqs,np.arange(len(freqs))):
            dm_shifted_template = apply_dm_delay(templates[f], dms[i], freq, n_points)
            modeled_signal = translate_signal(dm_shifted_template, diffs[i])
            band_signals.append(modeled_signal)
        modeled_signals.append(band_signals)
    
    return np.array(modeled_signals)  # Shape: (3 signals, 2 bands, n_points)

# Generate the datasets
datasets = generate_datasets()

x = np.linspace(0, 1, datasets.shape[2])

modeled_signals = model_signal(10,10,10, 0.10, 0.35, 0.7,datasets)

#plot_it(modeled_signals,datasets)


# Define the likelihood class
class SimpleGaussianLikelihood(bilby.Likelihood):
    def __init__(self, data, model, sigma):
        super().__init__(parameters={"dm0": None,"dm1": None,"dm2": None, "d1": None, "d2": None})
        self.data = data
        self.N = data.shape[2]
        self.model = model
        self.sigma = sigma

    def log_likelihood(self):
        est = self.model(self.parameters["dm0"],self.parameters["dm1"],self.parameters["dm2"], self.parameters["d0"], self.parameters["d1"], self.parameters["d2"],self.data)
        
        ll = 0
        for k in range(3):
            for band in range(len(freqs)):
                res = self.data[k, band] - est[k, band]
                ll += -0.5 * (
                    np.sum((res / self.sigma) ** 2) + self.N * np.log(2 * np.pi * self.sigma**2)
                )
        return ll

likelihood = SimpleGaussianLikelihood(data=datasets, model=model_signal, sigma=0.01)

#def negative_log_likelihood(params):
#    dm, d1, d2 = params
#    likelihood.parameters={"dm0": dm0, "dm0": dm1, "dm0": dm1, "d1": d1, "d2": d2}
#    return -likelihood.log_likelihood()
    
    
    
# Set up and run the sampler

priors = dict(
    
    dm0=bilby.core.prior.Uniform(9, 12, "dm0"),
    dm1=bilby.core.prior.Uniform(9, 12, "dm1"),
    dm2=bilby.core.prior.Uniform(9, 12, "dm2"),
    d1=bilby.core.prior.Uniform(0, 1, "d1"),
    d2=bilby.core.prior.Uniform(0, 1, "d2"),
)
priors['d0']=0

result = bilby.run_sampler(
    likelihood=likelihood,
    priors=priors,
    sampler="dynesty",
    nlive=250,
    outdir=outdir,
    label=label,
    resume = False,
    clean=True,
)
result.plot_corner(save=False)




# Perform the optimization
#result = minimize(
#    negative_log_likelihood,
#    initial_guess,
#    method="L-BFGS-B",  # A robust gradient-based optimizer
#    bounds=[(0,50) ,(0, 100), (0, 100)],  # Bounds on d1 and d2
#)

# Extract the optimized parameters
#optimal_dm, optimal_d1, optimal_d2 = result.x
#print(f"Optimal d1: {optimal_d1}, Optimal d2: {optimal_d2}")

# Use the optimized parameters to model the signals
#modeled_signals = model_signal(optimal_dm,0, optimal_d1, optimal_d2)


# Plot the results
modeled_signals = model_signal(result.posterior["dm0"].median(),result.posterior["dm1"].median(),result.posterior["dm2"].median(), 0, result.posterior["d1"].median(), result.posterior["d2"].median(),datasets)
print(result.posterior["dm0"].median(),result.posterior["dm1"].median(),result.posterior["dm2"].median(), 0, result.posterior["d1"].median(), result.posterior["d2"].median())
plot_it(modeled_signals,datasets)
