#fix for https://github.com/neuropsychology/NeuroKit/issues/593
def _hrv_dfa(peaks, rri, out, n_windows="default", **kwargs):
    if "dfa_windows" in kwargs:
        dfa_windows = kwargs["dfa_windows"]
    else:
        print(rri.shape)
        dfa_windows = [(4, rri.shape[0]-1), (rri.shape[0], None)]
        print(dfa_windows)
    
    # Determine max beats
    if dfa_windows[1][1] is None:
        max_beats = len(peaks) / 10
    else:
        max_beats = dfa_windows[1][1]
    
    # No. of windows to compute for short and long term
    if n_windows == "default":
        n_windows_short = int(dfa_windows[0][1] - dfa_windows[0][0] + 1)
        n_windows_long = int(max_beats - dfa_windows[1][0] + 1)
    elif isinstance(n_windows, list):
        n_windows_short = n_windows[0]
        n_windows_long = n_windows[1]
    
    # Compute DFA alpha1
    short_window = np.linspace(dfa_windows[0][0], dfa_windows[0][1], n_windows_short).astype(int)
    # For monofractal
    print('fractal_dfa')
    out["DFA_alpha1"] = fractal_dfa(rri, multifractal=False, windows=short_window, **kwargs)[0]
    # For multifractal
    mdfa_alpha1 = fractal_dfa(
        rri, multifractal=True, q=np.arange(-5, 6), windows=short_window, **kwargs
    )[1]
    out["DFA_alpha1_ExpRange"] = mdfa_alpha1["ExpRange"]
    out["DFA_alpha1_ExpMean"] = mdfa_alpha1["ExpMean"]
    out["DFA_alpha1_DimRange"] = mdfa_alpha1["DimRange"]
    out["DFA_alpha1_DimMean"] = mdfa_alpha1["DimMean"]
    
    # Compute DFA alpha2
    # sanatize max_beats
    if max_beats < dfa_windows[1][0] + 1:
        # warn(
        #     "DFA_alpha2 related indices will not be calculated. "
        #     "The maximum duration of the windows provided for the long-term correlation is smaller "
        #     "than the minimum duration of windows. Refer to the `windows` argument in `nk.fractal_dfa()` "
        #     "for more information.",
        #     category=NeuroKitWarning,
        # )
        return out
    else:
        long_window = np.linspace(dfa_windows[1][0], int(max_beats), n_windows_long).astype(int)
        # For monofractal
        out["DFA_alpha2"] = fractal_dfa(rri, multifractal=False, windows=long_window, **kwargs)[0]
        # For multifractal
        mdfa_alpha2 = fractal_dfa(
            rri, multifractal=True, q=np.arange(-5, 6), windows=long_window, **kwargs
        )[1]
        out["DFA_alpha2_ExpRange"] = mdfa_alpha2["ExpRange"]
        out["DFA_alpha2_ExpMean"] = mdfa_alpha2["ExpMean"]
        out["DFA_alpha2_DimRange"] = mdfa_alpha2["DimRange"]
        out["DFA_alpha2_DimMean"] = mdfa_alpha2["DimMean"]
    
    return out

