# Covid-Detection-Using-X-Ray-Images
Image recognition model that detects presence of covid based on x-ray images.

After installing the requirements or confirming that these are already installed and then cloning the repo, the data must be downloaded. Due to its large size and limitations on Github storage, images could be downloaded from https://www.kaggle.com/datasets/praveengovi/coronahack-chest-xraydataset?select=Coronahack-Chest-XRay-Dataset. The required metadata,however, is already made available in this repository.

The code could then be run. Originally there were no covid images in the test set, so I took a sample of the covid train images and included them in the test set. The remaining covid train images were randomly oversampled and augmented to compensate for class imbalance. Running the model training code will result in the creation of a tensorflow model object saved in a folder called "Image_Recog". This new folder must be in the same directory as the deployment code.

Overall test accuracy is 82.3%, while recall for Covid test cases is 92.9%.
