from distutils.core import setup

setup(name='wearablecompute',
      packages = ['wearablecompute'],
      version='0.1',
      description='Feature Engineering for Longitudinal Wearable Sensors',
      url='https://github.com/brinnaebent/wearablecompute',
      download_url = 'https://github.com/brinnaebent/wearablevar/archive/0.4.tar.gz',
      author='Brinnae Bent',
      author_email='runsdata@gmail.com',
      keywords = ['wearables', 'mhealth', 'features', 'feature engineering'],
      license='MIT',

      install_requires=['pandas','numpy','datetime', 'scipy',],

      classifiers=[
        'Development Status :: 3 - Alpha',      # Chose either "3 - Alpha", "4 - Beta" or "5 - Production/Stable" as the current state of your package
        'Intended Audience :: Developers',      
        'Topic :: Software Development :: Build Tools',
        'License :: OSI Approved :: MIT License',   
        'Programming Language :: Python :: 3',      
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        ],
)
