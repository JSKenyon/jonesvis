from jonesvis.jones.delay import Delay
from jonesvis.jones.diag_complex import DiagComplex
from jonesvis.jones.crosshand_delay import CrosshandDelay
from jonesvis.jones.bandpass import Bandpass
from jonesvis.jones.leakage import Leakage

JONES_TYPES = {
    "diag_complex": DiagComplex,
    "delay": Delay,
    "crosshand_delay": CrosshandDelay,
    "bandpass": Bandpass,
    "leakage": Leakage
}