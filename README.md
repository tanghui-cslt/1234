# GENEOs
This repository allows to reproduce the results in [Towards a topological-geometrical theory of group equivariant non-expansive operators for data analysis and machine learning](https://arxiv.org/abs/1812.11832).

## How to install
We recommend the use of virtual environments. In case you prefer not to use them skip the first point of the following list and the second line of the instruction at point 4.

1. Create a virtual environment for the project by typing in the terminal
    ```
    sudo pip install virtualenv
    sudo pip install virtualenvwrapper
    nano ~/.bashrc
    ```
    copy and paste at the end of the file
    ```
    export WORKON_HOME=$HOME/.virtualenvs
    export PROJECT_HOME=$HOME/Devel
    source /usr/local/bin/virtualenvwrapper.sh
    ```
    To understand something more or if the instructions do not work, visit [the virtualenvwrappers website](http://virtualenvwrapper.readthedocs.io/en/latest/install.html)

2. We assume you already installed git, python (>=3.5) and pip on your machine
3. Clone the repository by typing in the terminal

    ```
    git clone https://gitlab.com/mattia.bergomi/geneo.git
    ```
4. Continue with the installation by typing
    ```
    cd geneo
    mkvirtualenv --python=python3 geneo
    pip install -e ./
