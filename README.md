# Spacecraft Reentry Simulation

![Screenshot](/assets/Frame%20169.png)
![Screenshot](/assets/Screenshot%202023-04-26%20at%2014.34.10.png)

| | |
|:-------------------------:|:-------------------------:|
![Screenshot](/assets/Screenshot%202023-04-26%20at%2016.39.40.png) | ![Screenshot](+/../assets/newplot%20(1).png)
![Screenshot](/assets/newplot%20(2).png)  |  ![Screenshot](/assets/Screenshot%202023-04-26%20at%2019.28.15.png)

This is a Python-based web application that simulates the complex dynamics of a spacecraft orbiting around the Earth. It takes into account the Earth's rotation, J2 perturbations, atmospheric drag, and the Sun and Moon's gravity while predicting the spacecraft's trajectory.

The simulation uses the amazing [poliastro](https://docs.poliastro.space/en/stable/) library, as well as [astropy](https://www.astropy.org/) and [streamlit](https://streamlit.io/).

## Features

- User can define spacecraft initial state (position, velocity, azimuth, latitude, longitude, and altitude).
- User can set simulation parameters (start time, duration, and time step).
- Simulation results can be downloaded as CSV files.
- Visualization of spacecraft trajectory in 3D and ground track on a map.
- Displays altitude and velocity profiles over time.
- Detects and reports spacecraft reentry and impact.
- US Standard Atmosphere 1976 model is used to calculate atmospheric density.

## Dependencies

- astropy
- base64
- cartopy
- coordinate_converter (custom library)
- datetime
- numpy
- pandas
- plotly
- poliastro (mostly removed all references to this library, but still required as it is used to get keplerian orbit preview, but is not used in the simulation)
- streamlit
- numba (high performance JIT compiler for python that makes the simulation run faster)

## Usage

1. Install the necessary dependencies by running `pip install -r requirements.txt`.
2. Run the application with `streamlit run app.py`.
3. Open the provided URL in a web browser to access the application.

## Electron App

1. Install electron globally by typing `npm install -g electron` in a terminal window.
2. To run the electron app, open your terminal and run `./start.sh`. This will start the streamlit server and the electron app.
3. When you close electron it will kill the streamlit server.
4. Make sure you have the port `8501` available before running the script.

## Customization

You can modify the code to add more features or to change the existing behavior of the simulation. Some possible improvements include:

- Add support for different atmospheric models.
- Include more forces in the simulation, like solar radiation pressure or Earth's higher order gravitational harmonics.
- Implement a more accurate numerical integration method.

## License

This project is free and open-source. You are allowed to use, modify, and distribute the code as long as you give appropriate credit to the original creator.

## Acknowledgements

- [poliastro](https://docs.poliastro.space/en/stable/)
- [astropy](https://www.astropy.org/)
- [streamlit](https://streamlit.io/)
- [electron](https://www.electronjs.org/)
