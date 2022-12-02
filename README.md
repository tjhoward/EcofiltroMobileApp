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
 
