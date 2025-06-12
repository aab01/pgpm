import numpy as np

def makesignal(params):
    """
    Generates a synthetic signal with random structure and corresponding ground truth labels.

    This function creates a 1D signal of length `NTimeSamples` with multiple waveform patterns 
    (e.g., square, sine, cosine, ramps) and their corresponding ground truth (GT) labels. 
    The GT labels are provided as a one-hot encoded array for semantic segmentation tasks. 
    Additionally, each signal is assigned a "whole-signal" class based on specific characteristics 
    of the signal, such as overlapping frequency ranges of cosine waves.

    Args:
        params (dict): A dictionary containing the following keys:
            - 'NTimeSamples' (int): The length of the signal.
            - 'noise_level' (float): The level of noise to add to the signal.
            - 'randomoffset' (bool): Whether to use a random starting offset for the first waveform.
            - 'startoffset' (int): A fixed starting offset (used if 'randomoffset' is False).
            - 'mu_A' (float): Mean amplitude for the waveforms.
            - 'sigma_A' (float): Standard deviation of the amplitude for the waveforms.
            - 'num_segments' (int): Number of waveform segments in the signal.
            - 'with_replacement' (bool): Whether to sample waveform shapes with replacement.

    Returns:
        tuple: A tuple containing:
            - y (numpy.ndarray): The generated signal (1D array of shape `(NTimeSamples,)`).
            - patterntype (numpy.ndarray): One-hot encoded GT labels for waveform shapes 
                (2D array of shape `(NTimeSamples, 6)`).
            - signalclass (int): The whole-signal class (0 or 1).

    Notes:
        - The signal may not contains all waveform types, and the order and placement of the 
        possible types varies.
        - The function supports data for developing and testing both semantic 
        dense temporal segmentation and whole-signal classification tasks.
        - The "whole-signal" class is determined by the frequency range of the cosine waveform, 
        and by different amplitude distributions for the square waves,
        but perfect separation of the two classes can be impossible.
        - If `with_replacement` is False, waveform shapes are sampled without replacement, 
        ensuring each shape appears only once in the signal.
    """
    # Returns a signal with some random structure as a 1xNTimeSamples array, and
    # a GT label (6xNTimeSamples) comprising 6 channels, one channel for each 
    # possible waveform class that might exist at each discrete time point along
    # in the signal. The channels currently have values  of 0 if not that 
    # waveform class, and a value of 1 for the indices that
    # are between the start and end indices matching a particular waveform type
    # of finite support. So, if we consider *each* time point 
    # (on the dim=1 axis, for each i in [0 NTimeSamples-1]),
    # we have a one-hot encoding for the type of waveform that is present,
    # at least for the ground truth data.

    # In addition to supporting semantic segmentation, there is a "whole-signal" 
    # class assigned to each full NTimeSamples length signal; the distinction between these
    # two classes is a portion of each signal where a short cosine wave occurs: the
    # two classes have different overlapping frequency ranges for the cosine. Thus, 
    # perfect separation of these two signal classes is impossible.

    # The signals can be used for training and testing the equivalent of 
    # "semantic segmentation" networks for time-series data, *in addition*
    # to testing whole-signal classification networks. Further, beecause of the
    # provision of the waveform shape in the form of a one-hot encoding per
    # time point, binary cross entropy can be used in training the network.

    # Semantic segmentation: if the signal is x, the network would  
    # learn a mapping from x \in R^{NTimeSamples} to y \in $([0,1]^6)^{NTimeSamples}$
    # Classification: if the signal is x, the network learns a mapping
    # from x \in R^{NTimeSamples} to {0,1}, i.e. 1 of two classes per signal.


    # Note that in the first version of this code, the signal was always 
    # 256 time points long, and this was used for the DSP 2025 paper
    NTimeSamples = params['NTimeSamples']
    noise_level = params['noise_level']


    if params['randomoffset']:
        startoffset = np.random.randint(low=1,
                                         high=round(NTimeSamples/params['hyperparams']['StartOffsetDilator']), 
                                         dtype=int)
    else: # Simpler version - not recommended, but here for backward compatability
        startoffset = params['startoffset']

    # Heuristically found to be good option to prevent signals either overlapping
    # each other (which can be a physical impossibility in some contexts), and also
    # to ensure there is something that can be reasonably detected.
    maxseglength = int(round((NTimeSamples - startoffset - 1)/params['hyperparams']['LengthDilator']))
    minseglength = int(round(maxseglength/params['hyperparams']['MaxMinSegRatio']))
    
    # Below, the 6 non-zero waveform shapes that might be found in each example
    def square(length, amp, signalclass): # Square wave
        # Note that the overall signal class manifests partly in the amplitude
        # of the square wave, and partly in an additive offset of the square wave
        delta = params['hyperparams']['AmpDeltaSigma']*float(signalclass)-params['hyperparams']['AmpDeltaSigma']/2.0
        return (amp+delta)*np.ones(length)
    
    def sine(length, amp, signalclass): # Sine wave, partial
        return amp*np.sin(np.pi*np.linspace(0,1,length))

    def uramp(length, amp, signalclass):  # One sided ramp
        return amp*np.linspace(0, 1, length)
    
    def dramp(length, amp, signalclass):  # One sided ramp
        return amp*np.linspace(1, 0, length)

    def cosine(length, amp, signalclass): # Cosine wave
        #Signal class adds an offset the frequency of the cosine wave
        delta = params['hyperparams']['FreqDeltaSigma']*np.random.randn() + (float(signalclass)-0.5) 
        return amp*np.cos(np.pi*(1+delta)*np.linspace(0,1,length)) 
        
    def udramp(length, amp, signalclass): # Up down ramp
        if (length % 2) == 1:
            length += 1
    
        a=np.linspace(0, amp, int(length/2)) # Ramp going up...
        b=np.linspace(amp, 0, int(length/2)) # Ramp coming down
        
        return np.concatenate((a,b))

    def getpattern(choice, seglength, signalclass):
        switcher = {
         1: square,
         2: sine,
         3: uramp,
         4: dramp,
         5: cosine,
         6: udramp
        }
        option = switcher.get(choice, lambda: np.zeros(seglength))
        segment = option(seglength, params['mu_A']+params['sigma_A']*np.random.randn(), signalclass)
        
        return segment
    
    def pick(Choices):
    
        x = np.random.permutation(Choices)
        selected = x[0]
        
        return selected
    
    # The main code to generate the signals
    n = np.linspace(0,1,256) # Create a time-base for the signals
    y = np.zeros_like(n) # Empty array in which to slot waveforms
    Choices = [1,2,3,4,5,6] # The possible signal waveforms that make up each signal

    # In addition to variations in waveform shape for each example,
    # there are two different classes of signal, typified by a difference
    # that is present in one of the waveforms
    signalclass = np.random.choice([0,1]) 
    patterntype = np.zeros((len(y),len(Choices))) # One-hot encoding for waveform shape

    # Note: each example signal *never* contains *all* the types of waveform shape. 
    # This prevents the network being able to fully predict the waveform
    # shape at each time point, avoiding a collapse of the difficulty
    # of the task to a trivial one. Even so, the network must learn to predict the
    # waveform class at later points in time given previous points in time. This is not
    # relevant if all time points are looked at simultaneously, but becomes
    # quite important if the data enter a classifier system as a stream of data.
    # Indeed, the behaviour of a network that takes streaming input data, 
    # such as one for real-time use, could be expected to base its decisions on
    # receiving the $k^{th}$ input on decisions for $k-1, k-2, k-3...$, where 
    # such decisions are based on a sliding window along time, indexed by $k$.
    # So, this dataset provides a framework to investigate such ("non-Markovian")
    # behaviour in a controlled way. 
    #
    # The waveform selection at each "slot" iteration can be done either
    # with or without replacement.
    
    for n in range(params['num_segments']):
        # Define the length of signal segments, uniformly selected between
        # minseglength and maxseglength
        seglength = np.round((maxseglength-minseglength)*np.random.rand())+minseglength
        seglength = seglength.astype(int)

        # Select one of the waveform shapes from the pool
        Choice = pick(Choices)

        # Generate the waveform in a little 1D array of length seglength
        segment = getpattern(Choice, seglength, signalclass)

        if not params['with_replacement']:
            Choices.remove(Choice) # Sampling without replacement
        

        # Slot the generated waveform into the signal vector, y
        startloc = n*maxseglength+startoffset
        endloc = min(startloc+len(segment),len(y))

        # If the segment is longer than the remaining space in y,
        # truncate it to fit (hence the min() function above) and the 
        # indexing below
        y[startloc:endloc] = segment[0:endloc-startloc]
        
        # Update the "Waveform Shape Channel" accordingly for that segment
        patterntype[startloc:endloc,Choice-1] = 1
    
    y = y + noise_level*np.random.randn(1,len(y))
        
    return y, patterntype, signalclass 

def generate_data(params, num_samples=1):

    signals = np.zeros((num_samples, params['NTimeSamples']))
    masks = np.zeros((num_samples, params['NTimeSamples'], 6))
    signalclass = np.zeros(num_samples, dtype=int)
    
    for c in range(num_samples):
        # Generate a signal with the specified parameters
        # and store it in the signals array. The signal generation p
        # arameters are passed as a dictionary to the makesignal function
        s, gt, sc = makesignal(params)
        
        signals[c,:] = s
        masks[c,:,:] = gt
        signalclass[c] = sc
        
    return signals, masks, signalclass

def get_params_from_json(json_filename):
    import json
    """
    Load parameters from a JSON file and store them in a dictionary.

    Args:
        json_filename (str): Path to the JSON file.

    Returns:
        dict: Dictionary containing the parameters.
    """
    with open(json_filename, 'r') as f:
        params = json.load(f)
    return params

