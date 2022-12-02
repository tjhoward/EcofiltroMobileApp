# EcofiltroMobileApp

This repository holds the mobile application code of the Ecofiltro app. This app provides information on water safety and allows users to upload pictures to a server to be classified as dirty or clean.

To run the mobile app do the following:
-Open android studio.
-Connect your android device via usb or emulator.. 
-Open the terminal, and navigate to the projects root directory.
-Verify React-Native 0.63.3 is installed since the app runs on React-Native
-Run the command “run-android --variant=release”



The server/src folder holds the server component of the app. The app runs the machine learning model on a Google Cloud Run server. The server first takes in an input image that the user sends. The image is then broken into blob crops, and lastly the machine learning model analyzes the blobs  and predicts if the water is clean or not. The result of the prediction is then returned to the user in JSON format, where the app will display one of three different screens based on the result.

To send requests to the server:
    -Make a POST request to "https://flaskcontainer-ymlclai44q-uw.a.run.app/image"
     with base64 image data in a header of any key name.
