import 'bootstrap/dist/css/bootstrap.min.css';
require('../css/app.css');

import * as tf from '@tensorflow/tfjs';

async function init() {
    let model;
    const num_digits = 24;
    const inputDom = document.querySelector('#input');
    const submitDom = document.querySelector('#submit');
    const resultDom = document.querySelector('#result');
    const modelDom = document.querySelector('#model');

    await tf.loadLayersModel('model/model.json').then(function(data) {
        model = data;
        model.summary(80, 1, x => {
            var li = document.createElement('li');
            li.textContent =  x;
            li.setAttribute('class', 'list-group-item');
            modelDom.prepend(li)
        });
        predict_some_numbers();
    });

    function bin_encode(n) {
        const a = [];
        var b = parseInt(n, 10).toString(2).split("");
        for (var i = 0; i < num_digits; i++) {
            a.push(((b[i] != undefined) ? parseInt(b[i]) : 0));
        }
        return a;
    }

    function predict(n) {
        var class_names = ['NAN', 'BUZZ', 'FIZZ', 'FIZZBUZZ'];
        var tensor = tf.tensor([bin_encode(n)]);
        var y = model.predict(tensor);
        var prediction = tf.argMax(model.predict(tensor).arraySync()[0]).arraySync();
        var result = (prediction > 0) ? class_names[prediction] : n;
        return result;
    }

    function predict_some_numbers() {
        for (var i = 0; i < 101; i++) {
            prepend_result(i, predict(i));
        }
    }

    function prepend_result(n, prediction) {
        var li = document.createElement('li');
        li.textContent =  n + ': ' + prediction;
        li.setAttribute('class', 'list-group-item');
        resultDom.prepend(li)
    }

    submitDom.addEventListener('click', function(e) {
        e.stopPropagation();
        e.preventDefault();
        var prediction = predict(parseInt(inputDom.value));
        prepend_result(inputDom.value, prediction);
    })
}

document.addEventListener('DOMContentLoaded', init);

