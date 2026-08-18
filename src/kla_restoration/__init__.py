"""KLA restoration package.

Restoration of degraded semiconductor inspection images: speckle noise
removal, Gaussian-blur deconvolution and 2x super-resolution in a single
forward pass.

Submodules are imported explicitly by callers, e.g.

    from kla_restoration.model import NAFNetLiteSR

rather than re-exported here, so that importing the package does not pull
in torch/cv2 for callers that only need one submodule.
"""
__version__ = "0.1.0"
