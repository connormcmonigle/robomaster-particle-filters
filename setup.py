from skbuild import setup

__version__ = "1.0.0"

base_setup_options = {
    "name": "robomaster_particle_filters",
    "version": __version__,
    "author": "Connor McMonigle",
    "author_email": "connormcmonigle@gmail.com",
    "description": "A native module implementing CUDA accelerated particle filters for the RoboMaster robotics competition.",
    "long_description": "",
    "zip_safe": False,
    "packages": ['robomaster_particle_filters'],
    "package_data": {'robomaster_particle_filters': ['__init__.pyi']},
    "package_dir": {'': 'src'},
    "python_requires": ">=3.8",
}

additional_native_setup_options = {
    "cmake_install_dir": 'src/robomaster_particle_filters',
}

setup(
    **base_setup_options,
    **additional_native_setup_options,
)
