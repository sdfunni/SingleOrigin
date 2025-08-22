"""Internal function for detecting if running in Jupyter."""

def is_running_in_jupyter():
    try:
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            return True  # Jupyter Notebook or JupyterLab
        elif shell == 'SpyderShell':
            return False  # Spyder
        else:
            return False  # Other environment
    except NameError:
        return False  # Not in IPython environment
