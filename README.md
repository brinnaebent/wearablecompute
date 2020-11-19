# wearablevar

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

wearablevar is a Python package that calculates wearable variability metrics from longitudinal wearable sensors. These features can be used as part of your feature engineering process.

wearblevar is part of the Digital Biomarker Discovery Pipeline, available at dbdp.org.


### Installation

wearablevar requires the pandas, numpy, and datetime packages.

Recommended: Install via pip:

```sh
$ pip install wearablevar
```

Install via git:

```sh
$ pip install git+git://github.com/brinnaebent/wearablevar.git
$ git clone
```

### Functions

Dillinger is currently extended with the following plugins. Instructions on how to use them in your own application are linked below.

| Plugin | README |
| ------ | ------ |
| summarymetrics | interday mean, median, minimum, maximum, Q1, Q3 |
| interdaycv | interday coefficient of variation |
| interdaysd | interday standard deviation |
| intradaycv | intraday coefficient of variation (mean, median, standard deviation) |
| intradaysd | intraday standard deviation (mean, median, standard deviation) |
| intradaymean | intraday mean (mean, median, standard deviation)|
| TIR | Time in Range (SD default=1), *Note time relative to SR |
| TOR | Time outside Range (SD default=1), *Note time relative to SR |
| POR | Percent Outside Range (%) (SD default=1) |
| MASE | Mean Amplitude of Sensor Excursions (SD default=1) |
| importe4 | Import sensor data in 2 columns: datetime type, sensor type  |
| importe4acc | Import tri-axial accelerometry data in 4 columns: datetime type, sensor type x,y,z  |


### Continued Development

We are frequently updating this package with new functions and insights from the DBDP (Digital Biomarker Discovery Pipeline). For more details on contributing your own functions to this package, see dbdp.org. 


License
----

MIT



[//]: # (These are reference links used in the body of this note and get stripped out when the markdown processor does its job. There is no need to format nicely because it shouldn't be seen. Thanks SO - http://stackoverflow.com/questions/4823468/store-comments-in-markdown-syntax)


   [dill]: <https://github.com/joemccann/dillinger>
   [git-repo-url]: <https://github.com/joemccann/dillinger.git>
   [john gruber]: <http://daringfireball.net>
   [df1]: <http://daringfireball.net/projects/markdown/>
   [markdown-it]: <https://github.com/markdown-it/markdown-it>
   [Ace Editor]: <http://ace.ajax.org>
   [node.js]: <http://nodejs.org>
   [Twitter Bootstrap]: <http://twitter.github.com/bootstrap/>
   [jQuery]: <http://jquery.com>
   [@tjholowaychuk]: <http://twitter.com/tjholowaychuk>
   [express]: <http://expressjs.com>
   [AngularJS]: <http://angularjs.org>
   [Gulp]: <http://gulpjs.com>

   [PlDb]: <https://github.com/joemccann/dillinger/tree/master/plugins/dropbox/README.md>
   [PlGh]: <https://github.com/joemccann/dillinger/tree/master/plugins/github/README.md>
   [PlGd]: <https://github.com/joemccann/dillinger/tree/master/plugins/googledrive/README.md>
   [PlOd]: <https://github.com/joemccann/dillinger/tree/master/plugins/onedrive/README.md>
   [PlMe]: <https://github.com/joemccann/dillinger/tree/master/plugins/medium/README.md>
   [PlGa]: <https://github.com/RahulHP/dillinger/blob/master/plugins/googleanalytics/README.md>
