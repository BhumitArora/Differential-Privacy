# AdaPDP: Adaptive Personalised Differential Privacy

An improved version of PDP mechanisms to increase the accuracy of query results when users have different privacy needs.

## Table of Contents

- [Overview](#overview)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgements](#Acknowledgements)

## Overview

Traditional Differential privacy adding mechanisms were giving poor results when users have different privacy requirements. Through AdaPDP, we can overcome this problem and get good results on different queries like mean, median, count, logistic regression etc. AdaPDP mainly selects the best noise-generating algorithm for a given query and performs multiple rounds of sampling, known as a utility-aware sampling mechanism, to maximise the privacy budget usage the individuals allow.

## Requirements

Python 3.11 environment

- numpy
- pandas
- scikit-learn
- matplotlib
- joblib
- scipy

## Installation

create and activate a virtual environment using the following command in the project directory:

- creation:
- python3 -m venv <env-name>

- activation:
- for mac:
- source <env-name>/bin/activate
- for windows:
- <env-name>\Scripts\activate

- Then install all the packages using:
- pip install package_name

## Usage

This code was tested on macbook and was working fine. If you are a windows users do the required changes accordingly. If there is no problem in the installation of packages then you just need to run all the cells of logisticregress.ipynb file.

## Contributing

I found the results on logistic regression query because the paper had relevant results for this only but if you are interested you can find results on other queries like mean, median and count. 

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgements

[AdaPDP](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9488825)
