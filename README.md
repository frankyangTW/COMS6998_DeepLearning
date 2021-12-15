# Empirical study of effective model pruning in various tasks
## Team members: Hariharan Jayakumar (hj2559), Hsuan-Ru Yang (hy2713)

### Project Description
Deep learning models have become increasingly huge and achieved impressive results in various tasks. In this study, we seek to determine the contribution of the lowest-performing weights in the performance of the model. We do this by dropping the weights from the model and evaluating the change in performance. If the dropping of weights does not result in any performance impact, this can serve as a good backdrop to come up with more efficient models.

We aim to conduct this study by selecting 2-4 popular tasks and 2-4 popular model architectures. We started with simpler tasks, e.g. classification and regression on 1D input data, and then experimented on complexer tasks like 2D image classification.

First, we begin by training the models on these tasks and evaluating their performance on a test set. Then we run our algorithm to prune the model and fine-tune it, before noting its performance again on the same test set. We will compare the performance of the model before and after pruning, and before and after fine-tuning.

### Repository Description
This repository contains code for our experiements.

### Sample Commands
See the jupyter notebooks to run the experiments.

### Results
![](/images/res1.png)
![](/images/res2.png)
![](/images/res3.png)
![](/images/res4.png)
![](/images/res5.png)
![](/images/res6.png)

## Results and Insights
* Fine-tuning is a crucial step after model pruning
* L1/L2 pruning outperforms random pruning
* Pruning FC layers is more effective than pruning CONV layers
* Unstructured pruning is more robust to pruning


