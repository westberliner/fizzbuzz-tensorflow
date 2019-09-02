# Fizz Buzz solved by a Neural Network in Tensorflow 2

A little experiment to get familiar with tensorflow 2.0 and tensorflow.js
Based on this [blogpost by Joel Grus](https://joelgrus.com/2016/05/23/fizz-buzz-in-tensorflow/).

## Requirements

* Python 3.6
* Pipenv
* Node 10

## Installation

```
$ pipenv install
$ cd html && npm i
```

## Create Model

```
$ pipenv run python train.py
```

## Show log

```
$ tensorboard --logdir=logs
```

## Test and export for Tensorflow.js

```
$ pipenv run python test.py
```

## Run or build html example

```
$ cd html
$ npm start
$ npm run-script build
```


