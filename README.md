# TEM GUI LESC

## Steps to create the Windows executable file

1. Clone the repository into a directory of your choice

```
git clone https://github.com/ganyguru/TEMGUILESC.git
```

2. Change directory into the project directory

```
cd TEMGUILESC
```

3. Create a Python virtual environment

```
python -m venv .venv
```

4. Activate the virtual environment

```
.venv\Scripts\activate.bat 
```

5. Install the packages required for the project

```
pip install -r requirements.txt
```

6. Run the command below to generate the windows executable

```
pyinstaller --add-data "dm3;dm3" --add-data "database.csv;." --add-data "FFTmodelv1.hdf5;." --add-data "logo.png;." --onefile --noconsole GUI_Final.py
```

7. Navigate to the `dist` directory in the project folder to see the executable file.
