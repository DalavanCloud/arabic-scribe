Scribe: Realistic Handwriting in Tensorflow
=======
See [original project](https://github.com/greydanus/scribe)


Samples
--------
* "project" :
 ![Sample output 1](static/iter-125000-l-project.png?raw=true)
* "mahmoud" :
 ![Sample output 2](static/iter-125000-l-mahmoud.png?raw=true)
* "nourhan" :
 ![Sample output 3](static/iter-125000-l-nourhan.png?raw=true)
* "karim" :
 ![Sample output 4](static/iter-125000-l-karim.png?raw=true)
* "learning" :
 ![Sample output 5](static/iter-125000-l-learning.png?raw=true)
* "machine" :
 ![Sample output 6](static/iter-125000-l-machine.png?raw=true)

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
* run the sampler in bash: `mkdir -p ./logs/figures && python run.py --sample --tsteps 360 --tsteps 24 --text "project" --bias 5`


About
--------
This model is trained on the [IAM online handwriting dataset](http://www.fki.inf.unibe.ch/databases/iam-on-line-handwriting-database) and was inspired by the model described by the famous 2014 Alex Graves [paper](https://arxiv.org/abs/1308.0850). It consists of a three-layer recurrent neural network (LSTM cells) with a Gaussian Mixture Density Network (MDN) cap on top.

The model at one time step looks like this

![Rolled model](static/model_rolled.png?raw=true)

Unrolling in time, we get
![Unrolled model](static/model_unrolled.png?raw=true)

Distributed graves model on two machines. TO run the training :
On the first machine :
>> python run.py --train --ps_hosts=<your first machine ip> --worker_hosts=<your second machine ip> --task_index=0 --job_name=ps
On the second machine :
>> python run.py --train --ps_hosts=<your first machine ip> --worker_hosts=<your second machine ip> --task_index=0 --job_name=worker

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
 * text : project
 * tsteps_per_ascii : 21
 * tsteps : 360
 * bias : 5
 * character window : ![sample 1 window](static/iter-125000-w-project.png?raw=true)
 * heatmap : ![sample 1 heatmap](static/iter-125000-g-project.png?raw=true)
2. Sample 2 :
 * text : mahmoud
 * tsteps_per_ascii : 30
 * tsteps : 360
 * bias : 5
 * character window : ![sample 2 window](static/iter-125000-w-mahmoud.png?raw=true)
 * heatmap : ![sample 2 heatmap](static/iter-125000-g-mahmoud.png?raw=true)
3. Sample 3 :
 * text : nourhan
 * tsteps_per_ascii : 25
 * tsteps : 360
 * bias : 5
 * character window : ![sample 3 window](static/iter-125000-w-nourhan.png?raw=true)
 * heatmap : ![sample 3 heatmap](static/iter-125000-g-nourhan.png?raw=true)
4. Sample 4 :
 * text : karim
 * tsteps_per_ascii : 29
 * tsteps : 360
 * bias : 3
 * character window : ![sample 4 window](static/iter-125000-w-karim.png?raw=true)
 * heatmap : ![sample 4 heatmap](static/iter-125000-g-karim.png?raw=true)
5. Sample 5 :
 * text : learning
 * tsteps_per_ascii : 20
 * tsteps : 360
 * bias : 5
 * character window : ![sample 5 window](static/iter-125000-w-learning.png?raw=true)
 * heatmap : ![sample 5 heatmap](static/iter-125000-g-learning.png?raw=true)
6. Sample 6 :
 * text : machine
 * tsteps_per_ascii : 26
 * tsteps : 360
 * bias : 5
 * character window : ![sample 6 window](static/iter-125000-w-machine.png?raw=true)
 * heatmap : ![sample 6 heatmap](static/iter-125000-g-machine.png?raw=true)
