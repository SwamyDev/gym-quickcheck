from setuptools import setup, find_packages

setup(name='gym_quickcheck',
      version='0.0.0',
      packages=find_packages(include='gym_quickcheck'),
      install_requires=['gym'],
      extras_require={'test': ['pytest']})
