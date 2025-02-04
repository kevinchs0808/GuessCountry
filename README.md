# GuessCountry: LastFM Social Network's Country Prediction

![plot](./my_ray_gnn_bucket/result/lastfm_network_visualization.jpg)

## Background

In this project, the dataset contains a Graph Network where the nodes are the users (including artists) and the edges represent mutual followers relationships.

Each node has a feature which indicates the list of artists that each user liked.

But, it turns out that this feature can be converted into another type of edge of the graph network.

Hence, we have two types of edges which are "CONNECTED" (blue edges) and "LIKED" (green edges).

## Goal of The Project

For your information, each user has the country origin information.

And for this project, we would like to create a system which can predict the location of the origin.

In order to build this system, we will use Graph Neural Network that processes two types of edges.

Because we have two types of edges, we can parallalize the training process of the graph, by using Ray.

# Model Architecture

Here is the Deep Learning Model that will be used for this project:

![plot](./my_ray_gnn_bucket/result/System_Design.png)

## Model Performance

We trained the model using 50 epochs and here is the loss plot

![plot](./my_ray_gnn_bucket/result/train_test_loss.jpg)

Here is the confusion matrix

![plot](./my_ray_gnn_bucket/result/confusion_matrix.jpg)

## Reasoning on Model Result

Based on the confusion matrix, it seems that the model is most capable of predicting correctly for the country (0, 6, 10, 17).

But, one potential reason why this happened is because the data contains more nodes with this country labels.

![plot](./my_ray_gnn_bucket/result/label_distribution.jpg)

## Scaling the GNN Training with Ray on EC2

Ray is a tool that can help to parallelize your model training. In this case, the model training can be parallelized based on two types of edges, which are "CONNECTED" and "LIKED".

========= Procedure to Setup Ray Cluster on EC2 and Parallelize GNN on Ray Cluster will be written soon =========

## Next Step

- Finetune the layers of the Neural Network Architecture to improve the model performance

## Citation

Source of Dataset: B. Rozemberczki and R. Sarkar. Characteristic Functions on Graphs: Birds of a Feather, from Statistical Descriptors to Parametric Models. 2020.

Book Title: Proceedings of the 29th ACM International Conference on Information and Knowledge Management (CIKM '20)

Pages: 1325–1334

Organization: ACM
