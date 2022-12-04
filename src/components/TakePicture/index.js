import React, { useState, useEffect } from 'react';
import { Text, View, TouchableOpacity } from 'react-native';
import { Camera } from 'expo-camera';
import styles from './styles';
import { Actions } from 'react-native-router-flux';
import * as Permissions from 'expo-permissions';
import * as MediaLibrary from 'expo-media-library';
import * as FS from 'expo-file-system';

uriToBase64 = async (uri) => {
  
  let base64 = await FS.readAsStringAsync(uri, {
    encoding: FS.EncodingType.Base64,
  });
  return base64;
};


const TakePicture = () => {
  const [hasPermission, setHasPermission] = useState(null)
  let camera;
  useEffect(() => {
    ; (async () => {
      const { status } = await Camera.requestPermissionsAsync()
      setHasPermission(status === 'granted')
    })()
  }, [])
  const __takePicture = async () => {
    if (!camera) return

    const options = { quality: 0.3, base64: true, scale: 0.5 };
    const photo = await camera.takePictureAsync(options);
    const source = photo.uri;
    if (source) {
      Actions.Processing();
      handleSave(photo); //previosuly was source
    
    }
  }
  const handleSave = async (photo) => {
    const { status } = await Permissions.askAsync(Permissions.CAMERA_ROLL);
    if (status === 'granted'){
      const assert = await MediaLibrary.createAssetAsync(photo.uri);
      // await MediaLibrary.createAlbumAsync("Ecofiltro", assert);//
      console.log("Picture saved");
      //console.log(photo.type);
      let base64 = await this.uriToBase64(photo.uri);
      //console.log(base64);

      await this.toServer({
        type: "image",
        base64: base64, //prev was photo.base64
        uri: photo.uri,
      });

     


    } else {
      console.log('No tienes permiso');
    }
  }
  if (hasPermission === null) {
    return <View />;
  }
  if (hasPermission === false) {
    return <Text>No access to camera</Text>;
  }
  return (
    <View
      style={{
        flex: 1
      }}
    >
      <Camera
        style={{ flex: 1 }}
        type={Camera.Constants.Type.back}
        ref={(r) => {
          camera = r
        }}
      >
        <View
          style={{
            flex: 1,
            backgroundColor: 'transparent',
            flexDirection: 'row'
          }}
        >
          <View
            style={{
              position: 'absolute',
              top: '5%',
              right: '5%'
            }}
          >
            <TouchableOpacity onPress={() => Actions.Evaluate(false)}>
              <Text
                style={{
                  color: '#fff',
                  fontSize: 20
                }}
              >
                X
              </Text>
            </TouchableOpacity>
          </View>
          <View
            style={{
              position: 'absolute',
              bottom: 0,
              flexDirection: 'row',
              flex: 1,
              width: '100%',
              padding: 20,
              justifyContent: 'space-between'
            }}
          >
            <View
              style={{
                alignSelf: 'center',
                flex: 1,
                alignItems: 'center'
              }}
            >
              <TouchableOpacity
                onPress={__takePicture}
                style={{
                  width: 70,
                  height: 70,
                  bottom: 0,
                  borderRadius: 50,
                  backgroundColor: '#fff'
                }}
              />
            </View>
          </View>
        </View>
      </Camera>
    </View>
  );
}

//send the data of the selected media file to the Flask server,
toServer = async (mediaFile) => {

  let content_type = "";
  let url = "https://flaskcontainer-ymlclai44q-uw.a.run.app/image"

  try {
    
    let response = await FS.uploadAsync(url, mediaFile.uri, {
      headers: {
        "sentImage": content_type,
      },
      httpMethod: "POST",
      uploadType: FS.FileSystemUploadType.BINARY_CONTENT,
    });

    

    console.log(response.headers);
    console.log(response.body);

    processResults(response.body)

  } catch (error) {
    console.log("Error with server!")
    Actions.Home();
  }


};

const processResults = (result) => {



  const obj = JSON.parse(result)
  const clean = obj.Prediction[0] * 100
  const dirty= obj.Prediction[1] * 100

  console.log(clean)
  console.log(dirty)

  if(clean >= 90){
    Actions.NegativeResult() //no bacteria
  }
  else if(dirty >= 90){
    Actions.PositiveResult() //found bacteria
  }
  else{
    Actions.InconclusiveResult()
    
  }

}


export default TakePicture;