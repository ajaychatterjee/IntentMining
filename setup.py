import setuptools

with open("README.md", "r") as fp :
    long_description = fp.read()

setuptools.setup(
    name='ShortTextClustering',
    version='1.0',
    author='Ajay Chatterjee',
    author_email='ajay.chatt03@gmail.com',
    description='ITER-DBSCAN Implementation for unbalanced short text and numerical data clustering',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/ajaychatterjee/IntentMining/',
    packages=['ShortTextClustering'],
    install_requires=['mpi4py>=2.0',
                      'numpy',
                      ],
    classifers=[
      "Programming Language :: Python :: 3",
      "Operating System:: OS Independent"
    ],
)