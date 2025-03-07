# jonesvis
A basic simulatation and imaging tool for demonstrating the effects of Jones terms on calibrator sources.

## Installation

`pip install jonesvis`

We recommend installing inside a virtual environment.


## Data

While you can point `jonesvis` at any Measurement Set, it is only designed for tiny ones. Two such datasets are available [here](https://github.com/JSKenyon/jonesvis/tree/main/data). Be sure to download the raw files.

## Usage

`jonesvis path/to/MS`

The above will start the `jonesvis` server and should print a message informing you of its address. This usually `localhost:5006`. If you navigate to this address in your webbrowser, you should be met with the application page. 

The `jonesvis` server can be killed using `Ctrl-C` in the terminal from which it was launched. 
