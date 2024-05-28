# Power User Prediction and Deploy To Google Kubernetes Engine (GKE)

## Power User Prediction Steps

In this project, power users were identified through a detailed process involving feature engineering, model selection, and evaluation. The methodology used for identifying power users is outlined below:

### 1. Feature Engineering
- **Data Augmentation**: Additional features were created, such as the average number of products in a basket, to enrich the dataset and improve model performance.
- **Target Column**: Initially, power users were identified based on a z-score method. However, this method resulted in very few power users, making it insufficient. Therefore, based on the distributions, users who spent more than $110 were identified as power users, forming the basis for the classification target.

### 2. Model Training and Oversampling
- **Training Dataset Preparation**: The training dataset was adjusted using various oversampling methods to handle class imbalances.
- **Oversampling Techniques**: Methods like RandomOverSample, SMOTE, and ADASYN were employed to balance the dataset, ensuring that the models could effectively learn from both the majority and minority classes.

### 3. Model Selection
- **Hyperparameter Optimization**: GridSearchCV was used for hyperparameter tuning to find the best model configurations.
- **Comparison of Models**: Several models were compared based on their performance metrics, including KNN, XGBoost, Logistic Regression, and Random Forest. The XGBoost model with ADASYN oversampling showed the best performance.


| Model                                       | Recall  | Precision | Log Loss |
|---------------------------------------------|---------|-----------|----------|
| KNN                                         | 51.65%  | 88.70%    | 6.88%    |
| KNN_RandomOverSample                        | 47.25%  | 23.50%    | 15.45%   |
| KNN_SMOTE                                   | 62.64%  | 19.40%    | 19.13%   |
| KNN_ADASYN                                  | 28.57%  | 35.60%    | 24.31%   |
| XGBoost                                     | 71.43%  | 100.00%   | 5.23%    |
| XGBoost_RandomOverSample                    | 71.43%  | 55.60%    | 2.70%    |
| XGBoost_SMOTE                               | 71.43%  | 97.00%    | 1.41%    |
| **XGBoost_ADASYN**                          | 71.43%  | 100.00%   | 1.33%    |
| Logistic Regression                         | 67.03%  | 89.70%    | 1.21%    |
| Logistic Regression_RandomOverSample        | 78.02%  | 10.50%    | 14.46%   |
| Logistic Regression_SMOTE                   | 76.92%  | 11.60%    | 12.64%   |
| Logistic Regression_ADASYN                  | 84.62%  | 6.40%     | 24.08%   |
| RandomForest                                | 70.33%  | 100.00%   | 4.49%    |
| RandomForest_RandomOverSample               | 63.74%  | 87.90%    | 5.12%    |
| RandomForest_SMOTE                          | 72.53%  | 42.00%    | 3.16%    |
| RandomForest_ADASYN                         | 74.73%  | 32.70%    | 4.43%    |


### 4. Threshold Adjustment and Model Evaluation
- **Threshold Tuning**: The threshold value of the model was adjusted to optimize performance. Despite testing various thresholds, the default value of 0.5 was retained as it provided the best balance between recall and precision.
- **ROC AUC Curve Analysis**: The performance of the XGBoost ADASYN model was further validated using the ROC AUC curve, demonstrating strong predictive capabilities.

### 5. Create and Push Docker Image to Docker Hub
    
    1.Build the Docker image using the Dockerfile provided. This command creates an image tagged as xgboost_adasyn_poweruser_image:v1.
    ```sh
    docker build -t xgboost_adasyn_poweruser_image:v1 .
    ```
    
    2.To verify that the image was built correctly, run it locally. This command runs the container in detached mode, mapping port 8000 of the container to port 8000 on the host.
    ```sh
    docker run -d -p 8000:8000 --name xgboost_container xgboost_adasyn_poweruser_image:v1
    ```
    
    3.Before pushing the image to Docker Hub, tag it appropriately with your Docker Hub username and repository name.
    ```sh
    docker tag xgboost_adasyn_poweruser_image:v1 yaseminbellioglu/xgboost_adasyn_poweruser_image:v1
    ```
    
    4.Finally, push the tagged image to Docker Hub. This step uploads the image to your Docker Hub repository, making it available for deployment on other machines.
    ```sh
    docker push yaseminbellioglu/xgboost_adasyn_poweruser_image:v1
    ```

## 6. Deploy to Google Kubernetes Engine (GKE)

Google Kubernetes Engine (GKE) provides a managed environment for deploying, managing, and scaling your containerized applications using Google infrastructure.

### 1. Enable the GKE API

1. Navigate to the [Google Kubernetes Engine](https://console.cloud.google.com/kubernetes) in the Google Cloud Console.
2. Click on "Enable" to enable the GKE API for your project.

### 2. Create an Autopilot Cluster

1. In the Google Cloud Console, go to the Kubernetes Engine section.
2. Click on "Create" and select "Autopilot" mode.
3. Configure the cluster settings such as the name, location, and other options as needed.
4. Click on "Create" to provision the Autopilot cluster. Google will manage the nodes and scaling for you.

### 3. Deploy the Docker Image to the Cluster

1. **Upload the Docker Image to Artifact Registry**:
    - Create an Artifact Registry repository if you don't already have one.
    - Push your Docker image to Artifact Registry using the following commands:
    ```sh
    gcloud auth login
    gcloud config set project PROJECT_ID
    gcloud auth configure-docker us-central1-docker.pkg.dev
    docker tag xgboost_adasyn_poweruser_image:v1 us-central1-docker.pkg.dev/PROJECT_ID/REPOSITORY/xgboost_adasyn_poweruser_image:cloudingv1
    docker push us-central1-docker.pkg.dev/PROJECT_ID/REPOSITORY/xgboost_adasyn_poweruser_image:cloudingv1
    ```

2. **Deploy to GKE**:
    - Navigate to the "Workloads" section in the Kubernetes Engine.
    - Click on "Deploy" and select "Container image".
    - Enter the container image URL from Artifact Registry (e.g., `us-central1-docker.pkg.dev/PROJECT_ID/REPOSITORY/xgboost_adasyn_poweruser_image:cloudingv1`).
    - Configure the deployment settings such as the number of replicas, resources, and other options as needed.
    - Click on "Deploy" to create the deployment in your Autopilot cluster.

### 4. Expose the Deployment

1. Navigate to the "Services & Ingress" section in the Kubernetes Engine.
2. Click on "Expose" for your deployment.
3. Configure the service settings, such as the port and type (e.g., External load balancer for public access).
4. Click on "Expose" to create the service. The service will provide an endpoint for accessing your application.

### 5. Access the API

1. In the "Services & Ingress" section, find the external IP address for your service.
2. Open a browser and navigate to the external IP address followed by the appropriate path (e.g., `http://EXTERNAL_IP/docs`) to access the FastAPI interface.

This setup allows you to deploy and manage your containerized application on Google Kubernetes Engine, leveraging the benefits of automatic scaling and managed infrastructure provided by Autopilot.

## deployment.png : deploy image 
## (app_fastapi.png) : fastApi ınterface
## cluster.png : kebernets autopilot cluster
