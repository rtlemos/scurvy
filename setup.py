import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="scurvy",
    version="0.0.1",
    author="Ricardo Lemos",
    author_email="rtl00@yahoo.com",
    description="Space filling curves",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/rtlemos/scurvy/",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: GPL-3.0",
        "Operating System :: OS Independent",
    ],
)

