# Priors in one model mapper must be updated by priors in another model mapper

class TestSourceOnlyPipeline(object):
    def test_prior_passing(self):
        pass
    """
    Pipeline 1:

    PURPOSE - Fit a source-only image (i.e. no lens light component)

    PREPROCESSING:

    - Mark the brightest regions / multiple images of the source.
    - Draw a circle tracing the source (Einstein Radius / centre)
    - Draw circle / ellipse for the mask.

    NOTES:

    Image: Observed image used throughout.
    Mask: Assume a large mask (e.g. 2") throughout - this value could be chosen in preprocessing.

    INITIALIZATION PHASES:

    1) Mass: SIE+Shear
       Source: Sersic
       NLO: LM

    2) Mass: SIE+Shear (priors from phase 1)
       Source: 'smooth' pixelization (include regularization parameter(s) in the model)
       NLO: LM

    2H) Hyper-parameters: All included in model (most priors broad and uniform, but use previous phase regularization as well)
        Mass: SIE+Shear (Fixed to highest likelihood model from phase 2)
        Source: 'noisy' pixelization
        NLO: MN

    MAIN PIPELINE:

    a) Mass: SPLE+Shear (priors from Init phase 2)
       Source: 'noisy' pixelization (Fixed to init 2H hyper-parameters)
    """
