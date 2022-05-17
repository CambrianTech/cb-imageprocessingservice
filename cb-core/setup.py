from setuptools import setup
from Cython.Build import cythonize

setup(name='cambrian',
      version='0.41',
      description='Cambrian core module',
      url='https://github.com/CambrianTech/cb-core',
      author='Cambrian',
      packages=['cambrian'],
      zip_safe=False,
      entry_points={
            "console_scripts": [
                  "cb-splitdataset=cambrian.cmd.splitdataset:main",
                  "cb-exportmodel=cambrian.cmd.exportmodel:main",
                  "cb-savedmodelinfo=cambrian.cmd.savedmodelinfo:main",
                  "cb-runsavedmodel=cambrian.cmd.runsavedmodel:main",
                  "cb-quantizecoreml=cambrian.cmd.quantizecoreml:main",
            ],
      },
      ext_modules = cythonize("cambrian/LineFunctions.pyx")
)