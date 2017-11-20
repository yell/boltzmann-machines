# Run from a virtual environment
Clone the repository
```bash
git clone https://github.com/monsta-hd/boltzmann-machines
cd boltzmann-machines/
```
Install virtualenv if needed
```bash
pip install virtualenv
```
Create virtual environment for the project
```bash
virtualenv .venv
```
Activate virtual environment
```bash
source .venv/bin/activate
```
Install the dependencies
```bash
pip install -r requirements.txt
```

## deactivate and delete
If you are done working in the virtual environment, run
```bash
deactivate
```
To delete virtual environment, simply delete the folder
```bash
rm -rf .venv
```
