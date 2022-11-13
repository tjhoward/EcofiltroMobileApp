import { Actions } from 'react-native-router-flux';
import { connect } from 'react-redux';
import { normalize } from '../../utils/normalize';
import { Text, View, Image, TouchableOpacity } from 'react-native';
import { URL } from '../../../configuration'
import React from 'react';
import styles from './styles'

const Processing = ({  }) => (
    <View style={styles.container} >
        <Text style={styles.welcomeText}>
            {'Processing Image...'}
        </Text>
        <Text style={styles.introText}>
            {'Please wait for results'}
        </Text>

    </View>
);

export default connect(
    state => ({

    }),
    dispatch => ({

    }),
)(Processing);
