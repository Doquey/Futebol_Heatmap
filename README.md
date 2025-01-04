
# Futebol_Heatmap

**Futebol_Heatmap** is a simple program for generating heatmaps of player positions on a football field. The program uses homography to project points onto a layout of a football field, making the heatmap more visually intuitive.

## How it works
The program processes a video captured with a football tactical camera and transfers the positions of the players onto a scale model of the field. Using homography, the player positions are mapped accurately to the layout of the field, and a heatmap is generated to visualize player movement.

## Dataset
The entire field setup and a significant portion of the data used to train the model were sourced from the Roboflow project: [Roboflow Sports Dataset](https://github.com/roboflow/sports).

