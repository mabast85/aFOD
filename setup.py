from setuptools import setup, find_packages
from qboot_v2 import __version__


setup(
	
	name='qboot_v2',
	version=__version__,
	description='QBOOT V2',
	author='Matteo Bastiani',
	install_requires=['nibabel', 'scipy', 'matplotlib', 'seaborn', 'PyPDF2'],
	packages=find_packages(),

)