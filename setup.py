from setuptools import setup, find_packages
from aFOD import __version__


setup(

	name='aFOD',
	version=__version__,
	description='aFOD estimation tool',
	author='Matteo Bastiani',
	install_requires=['nibabel', 'scipy', 'cvxopt', 'numpy', 'progressbar2'],
	packages=find_packages(),
    include_package_data=True,

)
