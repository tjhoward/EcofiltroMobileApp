# EcofiltroMobileApp

This repository holds the mobile application code of the Ecofiltro app. This app provides information on water safety and allows users to upload pictures of water to a server to be classified as dirty or clean by a machine learning model.

The latest stable version was made in Android. To run the mobile app do the following:

- Open android studio.

- Connect your android device via USB or an emulator. 

- Open the terminal, and navigate to the projects root directory.

- Verify React-Native 0.63.3 is installed since the app runs on React-Native

- Run the command 'yarn install' to make sure all dependenices are installed

- Run the command “npx run-android --variant=release” to start the app



The server/src folder holds the server component of the app. The app runs the machine learning model on a Google Cloud Run server. The server first takes in an input image that the user sends. The image is then broken into blob crops, and lastly the machine learning model analyzes the blobs  and predicts if the water is clean or not. The result of the prediction is then returned to the user in JSON format, where the app will display one of three different screens based on the result.

To send requests to the server:

- Make a POST request to "https://flaskcontainer-ymlclai44q-uw.a.run.app/image" with base64 image data.
- The request is processed in the app.py file, which resides within a docker environment.

To update the server with new changes:

- Download the Google Cloud SDK at https://cloud.google.com/sdk
- Follow the instructions to setup your Google Cloud environment: https://cloud.google.com/sdk/docs/install-sdk 
- Run the command "gcloud builds submit --tag gcr.io/ecofiltrowaterapp/flaskcontainer;" within the root of your project directory 
- Run the command "gcloud run deploy --image gcr.io/ecofiltrowaterapp/flaskcontainer;" within the root of your project directory  

The server uses a XGBoost model for classifying images. The file for the model is located at  server/src/app/modelXGBoost.h5. After a POST request is made the following occurs within the app.py file:
 - The base64 image data is converted to a jpg image in the image() method.
 - The classifyImage() method is called, which is where we will feed the image into the model.
 - The cropBlobs() method is called to generate blob crops from the image.
 - The model analyzes the blob crops and returns a prediction on whether the image contains bacteria or not.
 - The prediction is returned as a JSON object to the mobile app, and a result screen is displayed to the user.
 

 
