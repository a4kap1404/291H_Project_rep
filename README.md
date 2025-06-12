### Important Note
The repo has been revamped to make it easier for users to test different configurations (design and pdk). The old version of the repository can be found in the zip file "291H_Project2-main_old.zip".

### Instructions
Note: Before running a file, look briefly at top of each file for variables to adjust (paths, designs, processes...)

(1): Setup Repo
- See gen_floorplan_and_copy.sh to create neccesary directories, generate ofrs products that will be used for testing/evaluation, and copy those products to a local directory. This repo has many of the directories already created, but if an error occurs in attempting to read/write to location, it is probably because
either the directory/file does not exist and should be created using this script, or that the paths in the python files being run have not been set properly.
- Before moving on, for each configuration (i.e. design and process) ensure the proper directories have been created and files been imported
  - this means:
    - ofrs_deliv/{process}/{design}/{files...}
      - files:
        - 3_2_place_iop.odb     (needed for ML placement)
        - 3_5_palce_dp.odb      (needed for measuring custom HPWL estimation of ORFS placement)
        - 3_3_place_gp.log      (needed for timing estimation)
        - 3_4_place_resized.log (not used for timing estimation, so dont really need)
        - 3_5_place_dp.log      (needed for timing estimation)
        - 3_5_place_dp.json     (needed for ORFS estimation of ORFS placement)
    - odbs/{process}/{design}/
      - Will be used to store files produced by scripts
        - includes:
          - 3_3 and 3_5 .def and .odb files
          - ml_place_graph.pkl produced place_p2.py
          - odb_placement_cell_map.pkl: produced by place_p1.py and stored for place_p3.py
          - odb_placement.pkl: produced by place_p1.py and used as input to place_p3.py

(2): Training
- Generate Dataset using datagen_main.py
  - Will take a long time, scales O(n^2) with num of cells, but can adjust certain hyperparams to speed it up
- Then train model on said dataset using train.py
  - Can use GPU to speed this up

(3): Evaluation
- To Evaluate the model and get metrics, there are 2 ways depending on your setup. If:
- Have a environment that has CUDA enabled and ORFS
  - use place.sh (currently have to run one at a time per configuration)
- Have 2 different enviorments for CUDA and ORFS
  - use gen_conversions.sh (use bash) on ORFS machine to run place_p1.py, and transfer files
  - use gen_gpu_ml_placements.sh (use bash) on CUDA machine to run place_p2.py, and transfer files
  - use gen_hpwl_gpu_reports.sh (use bash) on ORFS to get final reports

#### Results
- Mine can be found in final_results.txt. Yours once fully generated will be at the bottom of the results.log file.

### Description
This is a diffusion model implementaion heavily based on the paper "Chip Placement with Diffusion Models". It works by taking in ideal synthetic placements as training data, and diffusing them using the standard DDPM method through a linear noise schedule. During sampling (inference) it uses a form of potential-based guidance to steer the model to place results according to the constraints of overlap (noted as legality in code) and hpwl. During sampling, as the model reverses the noise, a potential is computed using estimates of hpwl and overlap over the estimation of x0 (the predicted final placement based on the current time step), then the gradient is taken with respect to x_t, and used to converge to better distributions of results learned during training. The model itself consists of GATs and linear layers. The attention layers found in the orignal were removed due to limitations on node size. 

We also produced a synthetic data generator that produces a placement give certain constaints such as die area and density. It produces a placement, then generates a netlist (2-pin edges version) based on it. The model will model the hypergraph as a digraph, which does mean it can not represent multiple conncetions between the same node, so if a real life example is given with multiple connections between 2 cells, only 1 will remain before being sent to the model.

Our model takes in the initial x,y positions of the provided initial placement of cells, uses the relative positons of pins as edge features for the GAT's, and takes in the sizing of the cell's height and width. It also contains residuals, normalization layers, and non-linears.

As for the guidance potential, ours uses overlap of cells with macros, out-of-chip-area violations, and estimates of total HPWL to guide the placement.

As for handling the additional features of IO pins, the model currently fails to take these into account, even though the data generator produces macros placements and IO pins. The plan is to modify the potential funciton to using a similar overlap score as currently used, as well adding a congestion estimator too. For the IO pins, some sort of score could be geneated based on distance between it and its connected instaces. Also tail and head cell/macro/io type can be features added to the edge matrix.

### Guide
We recommend first generating small dataset, of N = 50. This should take a couple minutes if the models are small in size. Dataset generation seems to take quite a while if one were to try out realistically sized placements (cells size > 600), so keep that in mind. The tool has parameters that effect 

After playing around with the dataset, you can train your model, and feel free to try and see if you can can reduce the MSE significantly and consistently below 1. I am not sure about the reason for specifically, but keep in mind the model is meant to take in normalized results, where the x and y are centered, and the longest dimension of the cell is scaled to [-1, 1], as was done in the paper.

We trained our model on a dataset of size 100, where each placement was around ~600 cells. We than ran the model with a diffusion step count of 20, which is very small (standard for ddpm is 1000 steps), but does speed the runtime significantly, as the model has to perform inference per iteration of sampling. Try generating a sizable and diverse dataset, then train the model (dont do this for reproduction, just run train.py as is). Afterwards feel free deploying it real life examples. Keep in mind the odb files are produced in the local directory, so you will need to move them to the proper directory to then launch the rest. This involves going the main Makefile in openroad, and starting from after they produced "3_5_..." (legal placement).

### File Descriptions (First read Instructions)

GenTestAndTrain.mk: need to move this using gen_floorplan_and_copy.sh into ofrs/flow and use to generate ONLY up to 3_5, and stops before routing

gen_floorplan_and_copy.sh: create neccesary directories, generate ofrs products that will be used for testing/evaluation, and copy those products to a local directory (dont need to run entire file as a lot of files and directoies have been included in the repo. Just try generating synthetic data, training, and running, unless something breaks).

place.sh: runs entire evaluation, but works on a environment that uses CUDA and ORFS

gen_conversions.sh: (use bash) on ORFS machine to run place_p1.py, and transfer files
- use ORFS environment

gen_gpu_ml_placements.sh: (use bash) on CUDA machine to run place_p2.py, and transfer files
- use CUDA environment
- writes to gpu_ml_placements.log

gen_hpwl_gpu_reports.sh: (use bash) on ORFS to get final reports
- use ORFS environment
- writes to results.log

datagen_main.py: use to generate synthetic dataset

model.py: define denoising model architecture

model_utils.py: defines linear noise schedule

place_p1.py: openroad based python file; Use to grab "3_2_..."  (io placement), and convert into graph for ML model
- dont need to run directly
- (launch using "openroad -python -exit place_p1.py")

place_p2.py: pytorch based python file; Uses to deploy model, and saves result. Also measures inference time.
- dont need to run directly

place_p3.py: exports back to odb and measures hpwl
- dont need to run directly
- (launch using "openroad -python -exit place_p3.py")

report_og_placement.py: report custom_hpwl metric and orfs_hpwl metric

train.py: Launches training loop and saves file

train_utils.py: Defines part of training/inference including sampling

results.log: used to store final metrics

### Limitations
Our model is limited in a variety manners. The most glaring issue is the lack of clustering, leading to other limitations such as an innability to deal with huge node amounts. Due to this, the standard cell to standard cell legality function was been removed, though it does work, and code within the sampling function has only been commented out. This limits our ability to measure local density.

Another issue was the lack of reduction in the MSE during training, even with a small batch of simple models. Experiments showed the MSE very quickly reducing to a value of 1, but never going subsantially below it. This could imply architecutal limitation of the current model as it pertains to diffusion. 

Our model during inference can have issues producing cells that stay within the boundaries, even with guidance potential. To patch this, we deployed tanh() functions after each sampling iteration.

Our model operates on a Digraph inwhich its cells are modeled as nodes. This means it cannot model multiple edges between the same 2 cells.

Also, currently the pin locations are not taken account, but an implementation of a guidance potential that measures distance hpwl between pins and cell would not be difficult. Nevertheless given, the lack of convergence during training, it is dubious the implementation would have made a difference.
