from setuptools import setup, find_packages

setup(
    name="aclimate_resampling",
    version='v0.0.30',
    author="stevensotelo",
    author_email="h.sotelo@cgiar.com",
    description="Resampling module",
    url="https://github.com/CIAT-DAPA/aclimate_resampling",
    download_url="https://github.com/CIAT-DAPA/aclimate_resampling",
    packages=find_packages('src'),
    package_dir={'': 'src'},
    keywords='resampling aclimate',
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
    entry_points={
        'console_scripts': [
            'aclimate_resampling=aclimate_resampling.aclimate_run_resampling:main',
        ],
    },
    install_requires=[
        "attrs==23.1.0",
        "affine==2.4.0",
        "bokeh==3.2.2",
        "cdsapi==0.6.1",
        "certifi==2023.7.22",
        "charset-normalizer==3.2.0",
        "click>=7.0.0",
        "click-plugins==1.1.1",
        "cligj==0.7.2",
        "cloudpickle==2.2.1",
        "colorama==0.4.6",
        "contourpy==1.1.0",
        "dask==2023.1.0",
        "distributed==2023.1.0",
        "fsspec==2023.6.0",
        "idna==3.4",
        "importlib-metadata==6.8.0",
        "Jinja2==3.1.2",
        "locket==1.0.0",
        "lz4==4.3.2",
        "msgpack==1.0.5",
        "MarkupSafe==2.1.3",
        "numpy>=1.25.2",
        "packaging==23.1",
        "pandas==2.0.3",
        "partd==1.4.0",
        "Pillow==10.0.0",
        "psutil==5.9.5",
        "pyarrow==13.0.0",
        "pyparsing==3.1.1",
        "python-dateutil==2.8.2",
        "pytz==2023.3",
        "PyYAML==6.0.1",
        "requests>=2.31.0",
        "six==1.16.0",
        "snuggs==1.4.7",
        "sortedcontainers==2.4.0",
        "tblib==2.0.0",
        "toolz==0.12.0",
        "tornado==6.3.3",
        "tqdm>=4.65.0",
        "tzdata==2023.3",
        "urllib3==2.0.4",
        "xarray==0.20.2",
        "xyzservices==2023.7.0",
        "zict==3.0.0",
        "zipp==3.16.2",
        "rioxarray==0.14.1",
        "netCDF4==1.6.4",
    ]
)
