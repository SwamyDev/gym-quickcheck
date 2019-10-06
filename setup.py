from setuptools import setup

test_requirements = ['pytest']

setup(name='gym_quickcheck',
      version='0.0.0',
      install_requires=['gym'],
      tests_require=test_requirements,
      extras_require={'test': test_requirements})
