import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()
setuptools.setup(
    name = "GaussRF"
    version = "0.0.1",
    author = " K. Mpehle"
    author_email = "khaya.mpehle@gmail.com",
    description = "Simulation of 1-D and 2-D rectangular random fields"
    long_description = long_description,
    install_requires = ["numpy", "scipy"]
    long_description_content_type = "text/markdown",
    classifiers = [
        "Programming Language :: Python :: 3",
        "Licence :: OSI :: MIT License",
        "Operating System :: OS Independent",
        ],
    python_requires = '>=3.2',
    )
