# TEM GUI LESC

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A graphical user interface for processing and analyzing Transmission Electron Microscopy (TEM) images.

## Features

- Process multiple TEM file formats (.dm3, .dm4, .mrc, .emd, .ser)
- Custom d-spacing database population
- Adjustable processing parameters including slice selection and detection sensitivity
- Automated image segmentation and analysis
- Comprehensive output with visualization options

## Workflow

### 1. Setup and Installation

First, install the application by following the [installation instructions](#installation-instructions) below.

### 2. Prepare Your Data

Place your TEM data files (.dm3, .dm4, .mrc, .emd, .ser) in the `put_your_data_here` folder:

![Input Data Placement](images/Input_Data.png)
*Example: Placing a .dm4 file in the input folder*

### 3. Launch the Application

Run the executable to open the initial GUI window:

![Initial Window](images/Initial_window.png)
*The main interface of TEM GUI LESC*

### 4. Configure Processing Parameters

Click "Start Processing" to open the processing window where you can adjust parameters:

![Processing Window](images/Processing_window.png)
*Configure slice range, pixel size, detection sensitivity, and more*

### 5. View Results

After processing completes, results are saved in the `processed_files` folder:

![Processed Files Folder](images/Processed_Files.png)
*Output folder structure with processed data*

Inside each processed file folder, you'll find detailed results:

![Processed Files Structure](images/Processed_Files_in.png)
*Structure of the output folder for each processed file*

Each slice directory contains comprehensive analysis data:

![Slice Directory](images/Slice_directory.png)
*Contents of an individual slice directory with analysis results*

### 6. Console Output (Optional)

If you enable the console during installation or run as a Jupyter notebook:

![Console Output](images/Console_output.png)
*Example of console output during processing*

## Installation Instructions

### Option 1: Using the Pre-built Executable

1. Download the latest release from the [Releases](https://github.com/ganyguru/TEMGUILESC/releases) page
2. Extract the zip file to a location of your choice
3. Create two folders in the same directory as the executable:
   - `put_your_data_here` (for input files)
   - `processed_files` (for output)
4. Run the executable

### Option 2: Building from Source

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
   - Windows:
     ```
     .venv\Scripts\activate.bat
     ```
   - macOS/Linux:
     ```
     source .venv/bin/activate
     ```

5. Install the required packages
   ```
   pip install -r requirements.txt
   ```

6. Run the application directly
   ```
   python GUI_Final.py
   ```

7. (Optional) Generate a Windows executable
   ```
   pyinstaller --add-data "database.csv;." --add-data "FFTmodelv1.hdf5;." --add-data "logo.png;." --onefile --noconsole GUI_Final.py
   ```
   The executable will be created in the `dist` directory.

## Image Organization for GitHub

For proper image display in your README:

1. Create an `images` folder in your GitHub repository
2. Upload all workflow images to this folder
3. Reference them in the README using relative paths as shown above

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use this software in your research, please cite:

```
@software{TEMGUILESC,
  author = {Your Name},
  title = {TEM GUI LESC: A Graphical User Interface for Processing TEM Images},
  url = {https://github.com/ganyguru/TEMGUILESC},
  year = {2025},
}
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
