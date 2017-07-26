Scribe: Realistic Handwriting in Tensorflow
=======
See [original project](https://github.com/greydanus/scribe)

Changes to optimize and pretrained model:
-----------------------------------------
1. No changes.

Samples
--------
* "machine learning" :
 ![Sample output 1](static/iter-124500-l-machine_le.png?raw=true)
* "mahmoud" :
 ![Sample output 2](static/iter-124500-l-mahmoud.png?raw=true)
* "nourhan" :
 ![Sample output 3](static/iter-124500-l-nourhan.png?raw=true)
* "karim" :
 ![Sample output 4](static/iter-124500-l-karim.png?raw=true)
* "learning" :
 ![Sample output 5](static/iter-124500-l-learning.png?raw=true)
* "machine" :
 ![Sample output 6](static/iter-124500-l-machine.png?raw=true)

Jupyter Notebooks
--------
For an easy intro to the code (along with equations and explanations) check out these Jupyter notebooks:
* [inspecting the data](https://nbviewer.jupyter.org/github/greydanus/scribe/blob/master/dataloader.ipynb)
* [sampling from the model](https://nbviewer.jupyter.org/github/greydanus/scribe/blob/master/sample.ipynb)

Getting started
--------
* install dependencies (see below).
* download the repo
* navigate to the repo in bash
* run the sampler in bash: `mkdir -p ./logs/figures && python run.py --sample --text "your text here" --bias 5`


About
--------
This model is trained on the [IAM online handwriting dataset](http://www.fki.inf.unibe.ch/databases/iam-on-line-handwriting-database) and was inspired by the model described by the famous 2014 Alex Graves [paper](https://arxiv.org/abs/1308.0850). It consists of a three-layer recurrent neural network (LSTM cells) with a Gaussian Mixture Density Network (MDN) cap on top.

The model at one time step looks like this

![Rolled model](static/model_rolled.png?raw=true)

Unrolling in time, we get
![Unrolled model](static/model_unrolled.png?raw=true)

The model was trained with default values.


Dependencies
--------
* All code is written in python 2.7. You will need:
 * Numpy
 * Matplotlib
 * [TensorFlow 1.0](https://www.tensorflow.org/install/)
 * OPTIONAL: [Jupyter](https://jupyter.org/) (if you want to run sample.ipynb and dataloader.ipynb)

Sample Information
------------------
1. Sample 1 :
 * text : machine learning
 * tsteps_per_ascii : default(25)
 * tsteps : 400
 * bias : 5
 * character window : ![sample 1 window](static/iter-124500-w-machine_le.png?raw=true)
 * heatmap : ![sample 1 heatmap](static/iter-124500-g-machine_le.png?raw=true)
2. Sample 2 :
 * text : mahmoud
 * tsteps_per_ascii : default(25)
 * tsteps : 200
 * bias : 5
 * character window : ![sample 2 window](static/iter-124500-w-mahmoud.png?raw=true)
 * heatmap : ![sample 2 heatmap](static/iter-124500-g-mahmoud.png?raw=true)
3. Sample 3 :
 * text : nourhan
 * tsteps_per_ascii : default(25)
 * tsteps : 200
 * bias : 5
 * character window : ![sample 3 window](static/iter-124500-w-nourhan.png?raw=true)
 * heatmap : ![sample 3 heatmap](static/iter-124500-g-nourhan.png?raw=true)
4. Sample 4 :
 * text : karim
 * tsteps_per_ascii : default(25)
 * tsteps : 150
 * bias : 3
 * character window : ![sample 4 window](static/iter-124500-w-karim.png?raw=true)
 * heatmap : ![sample 4 heatmap](static/iter-124500-g-karim.png?raw=true)
5. Sample 5 :
 * text : learning
 * tsteps_per_ascii : default(25)
 * tsteps : 200
 * bias : 5
 * character window : ![sample 5 window](static/iter-124500-w-learning.png?raw=true)
 * heatmap : ![sample 5 heatmap](static/iter-124500-g-learning.png?raw=true)
6. Sample 6 :
 * text : machine
 * tsteps_per_ascii : default(25)
 * tsteps : 200
 * bias : 5
 * character window : ![sample 6 window](static/iter-124500-w-machine.png?raw=true)
 * heatmap : ![sample 6 heatmap](static/iter-124500-g-machine.png?raw=true)