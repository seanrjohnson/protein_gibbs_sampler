from glob import glob
from os.path import basename, splitext
from setuptools import setup, find_packages
# from distutils.core import setup, find_packages

requirements = ['torch', 'esm', 'pytest']

setup(
    name="pgen",
    version="0.0.0",
    description="Utilizing Pre-trained evolutionary scale models for proteins.",
    author="GaTech DL",
    url="",
    license="MIT",
    packages=find_packages('src'),
    package_dir={'': 'src'},
    # py_modules=[splitext(basename(path))[0] for path in glob('src/pgen/*.py')],
    install_requires=requirements,
    include_package_data=True,
    zip_safe=False,
    # python_requires='>=3.6'
)