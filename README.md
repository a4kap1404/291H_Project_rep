### Description
This is a diffusion model implementaion heavily based on the paper "Chip Placement with Diffusion Models". It works by taking in ideal synthetic placements as training data, and diffusing them using the standard DDPM method through a linear noise schedule. During sampling (inference) it uses a form of potential-based guidance to steer the model to place results according to the constraints of overlap (noted as legality in code) and hpwl. During sampling, as the model reverses the noise, a potential is computed using estimates of hpwl and overlap over the estimation of x0 (the predicted final placement based on the current time step), then the gradient is taken with respect to x_t, and used to converge to better distributions of results learned during training. The model itself consists of GATs and linear layers. The attention layers found in the orignal were removed due to limitations on node size. 

We also produced a synthetic data generator that produces a placement give certain constaints such as die area and density. It produces a placement, then generates a netlist (2-pin edges version) based on it. The model will model the hypergraph as a digraph, which does mean it can not represent multiple conncetions between the same node, so if a real life example is given with multiple connections between 2 cells, only 1 will remain before being sent to the model.

Our model takes in the initial x,y positions of the provided initial placement of cells, uses the relative positons of pins as edge features for the GAT's, and takes in the sizing of the cell's height and width. It also contains residuals, normalization layers, and non-linears.

As for the guidance potential, ours uses overlap of cells with macros, out-of-chip-area violations, and estimates of total HPWL to guide the placement.

As for handling the additional features of IO pins, the model currently fails to take these into account, even though the data generator produces macros placements and IO pins. The plan is to modify the potential funciton to using a similar overlap score as currently used, as well adding a congestion estimator too. For the IO pins, some sort of score could be geneated based on distance between it and its connected instaces. Also tail and head cell/macro/io type can be features added to the edge matrix.

### Guide
We recommend first generating small dataset, of N = 50. This should take a couple minutes if the models are small in size. Dataset generation seems to take quite a while if one were to try out realistically sized placements (cells size > 600), so keep that in mind. The tool has parameters that effect 

After playing around with the dataset, you can train your model, and feel free to try and see if you can can reduce the MSE significantly and consistently below 1. I am not sure about the reason for specifically, but keep in mind the model is meant to take in normalized results, where the x and y are centered, and the longest dimension of the cell is scaled to [-1, 1], as was done in the paper.

We trained our model on a dataset of size 100, where each placement was around ~600 cells. We than ran the model with a diffusion step count of 30, which is very small, but does speed the runtime significantly, as the model has to perform inference per iteration of sampling. Try generating a sizable and diverse dataset, then train the model. Afterwards feel free deploying it real life examples. Keep in mind the odb files are produced in the local directory, so you will need to move them to the proper directory to then launch the rest. This involves going the main Makefile in openroad, and starting from after 

### File Descriptions
datagen_main.py: use to generate synthesic dataset

model.py: define denoising model architecture

model_utils.py: defines linear noise schedule

old_scripts.sh: moves design/pdk instance from openroad (assumes openroad installed in "of") to location directories

(launch using "openroad -python -exit place_p1.py")
place_p1.py: openroad based python file; Use to grab "3_2_..."  (io placement), and convert into graph for ML model

place_p2.py: pytorch based python file; Uses to deploy model, and saves result

(launch using "openroad -python -exit place_p3.py")
place_p3.py: openroad

train.py: Launches training loop and saves file

train_utils.py: Defines part of training/inference including sampling

### Limitations
Our model is limited in a variety manners. The most glaring issue is the lack of clustering, leading to other limitations such as an innability to deal with huge node amounts. Due to this, the standard cell to standard cell legality function was been removed, though it does work, and code within the sampling function has only been commented out. This limits our ability to measure local density.

Another issue was the lack of reduction in the MSE during training, even with a small batch of simple models. Experiments showed the MSE very quickly reducing to a value of 1, but never going subsantially below it. This could imply architecutal limitation of the current model as it pertains to diffusion. 

Our model durign inference can have issues producing cells that stay within the boundaries, even with guidance potential. To patch this, we deployed tanh() functions after each sampling iteration after the first 30% of iterations.
