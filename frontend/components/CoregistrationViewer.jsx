import React, { useEffect, useState } from 'react'
import io from 'socket.io-client'
import { DEBOUNCE, POSITION, ROTATION, SCALE, SERVER_URL } from '../constants';
import CoregistrationViewerControls from './CoregistrationControls';
import { useDebouncedEffect } from '../hooks/useDebouncedEffect';
import Loader from './Loader';

const socket = io(SERVER_URL)

function CoregistrationViewer () {
  const [images, setImages] = useState([])
    const [positionTransform, setPositionTransform] = useState(null)
  const [rotationTransform, setRotationTransform] = useState(null)
  const [scaleTransform, setScaleTransform] = useState(null)
  const [loading, setLoading] = useState(false)

    socket.on('images', (base64Images) => {
    setLoading(false)
    setImages(base64Images)
    })

    useEffect(() => {
    // Send a "start" event to the server to start the image stream
    socket.emit('start')

        return () => socket.disconnect()
    }, [])

    const onControlsChange = (transform, axis, amount) => {
    switch (transform) {
      case SCALE:
        setScaleTransform({ axis, amount })
        break
            case ROTATION:
        setRotationTransform({ axis, amount })
        break
            case POSITION:
        setPositionTransform({ axis, amount })
        break
        }
  }

  useDebouncedEffect(() => {
    if (scaleTransform) {
      setLoading(true)
      socket.emit('transform', SCALE, scaleTransform.axis, scaleTransform.amount)
    }
  },
  [scaleTransform], DEBOUNCE)
    useDebouncedEffect(() => {
    if (rotationTransform) {
      setLoading(true)
      socket.emit('transform', ROTATION, rotationTransform.axis, rotationTransform.amount)
    }
  },
  [rotationTransform], DEBOUNCE)
    useDebouncedEffect(() => {
    if (positionTransform) {
      setLoading(true)
      socket.emit('transform', POSITION, positionTransform.axis, positionTransform.amount)
    }
  },
  [positionTransform], DEBOUNCE)

    return (
      <div style={{ display: 'flex', flexDirection: 'row', alignItems: 'center' }}>
          <Loader open={loading} />
          <CoregistrationViewerControls onChange={onControlsChange} />
          <div style={{ display: 'flex', flexDirection: 'row' }}>
              {images.map((base64Image, index) => (
                  <img
src={`data:image/jpeg;base64,${base64Image}`}
                      alt={`Image ${index + 1}`} key={index} style={{ maxWidth: `${100 / images.length}%` }}
                    />
                ))}
            </div>
        </div>

  )
}

export default CoregistrationViewer
